"""JSON-described private tools, bounded transforms and declarative compositions."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from pathlib import Path
from threading import RLock

from workflow_approvals import HumanCheckRequired
from workflow_basic_tools import TOOL_CONTRACTS, execute_basic
from workflow_declarative_tools import DeclarativeTools
from workflow_http_tools import HTTPPolicy
from workflow_nodes import NodeContext
from workflow_tool_interfaces import PrivateToolModule
from workflow_tool_safety import DEFAULT_TOOL_SAFETY_CONFIG, ToolSafetyPolicy, emit_security_event
from workflow_tools import ToolDefinition, require_tool_human_check
from workflow_types import JsonDict

DEFAULT_PRIVATE_TOOL_CATALOG = Path(__file__).parent / "tools" / "private_tools.json"
TEMPLATE_DIRECTORY = ".dgs-tools"


class BuiltinPrivateToolModule(PrivateToolModule):
    def __init__(
        self,
        workspace_root: str | Path,
        catalog_path: str | Path | None = None,
        *,
        safety_config_path: str | Path | None = None,
        on_security_event=None,
        http_allowed_origins: list[str] | None = None,
        http_credentials: dict[str, dict[str, str]] | None = None,
    ) -> None:
        if on_security_event is not None and not callable(on_security_event):
            raise TypeError("on_security_event must be callable.")
        self.on_security_event = on_security_event
        self._authorized_calls = {}
        self._template_lock = RLock()
        config = {"workspace_root": str(workspace_root)}
        if http_allowed_origins is not None:
            config["http_allowed_origins"] = http_allowed_origins
        if http_credentials is not None:
            config["http_credentials"] = http_credentials
        if safety_config_path is not None:
            config["safety_config_path"] = str(safety_config_path)
        if catalog_path is not None:
            config["catalog_path"] = str(catalog_path)
        self.configure(config)

    def configure(self, config: JsonDict) -> None:
        """Load/reload atomically. Root and resource limits are host settings."""
        unknown = set(config) - {
            "workspace_root",
            "catalog_path",
            "max_read_bytes",
            "max_write_bytes",
            "max_entries",
            "safety_config_path",
            "http_allowed_origins",
            "http_credentials",
            "declarative_allowed_tools",
        }
        if unknown:
            raise ValueError(f"Unknown private-tool settings: {sorted(unknown)}")
        root = Path(config["workspace_root"]).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("workspace_root must be an existing directory.")
        limits = {
            name: config.get(name, default)
            for name, default in (
                ("max_read_bytes", 262144),
                ("max_write_bytes", 262144),
                ("max_entries", 200),
            )
        }
        if any(type(value) is not int or value < 1 for value in limits.values()):
            raise ValueError("Private-tool limits must be positive integers.")
        catalog_path = Path(config.get("catalog_path", DEFAULT_PRIVATE_TOOL_CATALOG)).resolve(
            strict=True
        )
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        extra_fields = set(raw) - {"format_version", "tools"} if isinstance(raw, dict) else set()
        if extra_fields:
            raise ValueError(f"Unknown private tool catalog fields: {sorted(extra_fields)}")
        if (
            not isinstance(raw, dict)
            or raw.get("format_version") != "1.0"
            or not isinstance(raw.get("tools"), list)
        ):
            raise ValueError("Expected a version 1.0 private-tool catalog with a tools list.")
        catalog = {}
        for item in raw["tools"]:
            tool = ToolDefinition.from_dict(item)
            if tool.name not in TOOL_CONTRACTS:
                raise ValueError(f"No bundled implementation for tool {tool.name!r}.")
            if tool.name in catalog:
                raise ValueError(f"Duplicate tool name: {tool.name!r}")
            self._check_schema(tool)
            catalog[tool.name] = tool
        safety_path = Path(config.get("safety_config_path", DEFAULT_TOOL_SAFETY_CONFIG)).resolve(
            strict=True
        )
        safety = ToolSafetyPolicy(safety_path)
        origins = config.get("http_allowed_origins", [])
        if not isinstance(origins, list):
            raise ValueError("http_allowed_origins must be a list.")
        http_policy = HTTPPolicy(origins, config.get("http_credentials", {}))
        allowed = config.get(
            "declarative_allowed_tools", [name for name in catalog if name != "tools.define"]
        )
        if (
            not isinstance(allowed, list)
            or any(not isinstance(name, str) for name in allowed)
            or len(set(allowed)) != len(allowed)
            or not set(allowed) <= catalog.keys()
            or "tools.define" in allowed
        ):
            raise ValueError("Invalid declarative tool whitelist.")
        self._declarative_allowed_tools = frozenset(allowed)
        self.safety_config_path = safety_path
        self._safety = safety
        self.workspace_root = root
        self.catalog_path = catalog_path
        self._limits = limits
        self._catalog = catalog
        self._http = http_policy
        self._declarative = DeclarativeTools(self)
        self._authorized_calls.clear()

    @property
    def safety_policy(self) -> ToolSafetyPolicy:
        """The file safety policy loaded by the last ``configure`` call."""
        return self._safety

    @staticmethod
    def _check_schema(tool: ToolDefinition) -> None:
        required, optional = TOOL_CONTRACTS[tool.name]
        schema = tool.input_schema
        properties = schema.get("properties", {})
        declared_required = schema.get("required", [])
        if (
            not isinstance(properties, dict)
            or set(properties) != set(required) | set(optional)
            or not isinstance(declared_required, list)
            or any(not isinstance(name, str) for name in declared_required)
            or sorted(declared_required) != sorted(required)
            or schema.get("additionalProperties") is not False
        ):
            raise ValueError(f"Input schema does not match the bundled {tool.name!r} signature.")
        for name, expected in {**required, **optional}.items():
            if (
                not isinstance(properties[name], dict)
                or properties[name].get("type")
                != ({str: "string", bool: "boolean", dict: "object", list: "array"}[expected])
            ):
                raise ValueError(f"Incorrect argument type in {tool.name!r} schema: {name!r}")

    def list_tools(self) -> list[ToolDefinition]:
        return [self.get_tool(name) for name in self._catalog] + [
            self._declarative.tool(d) for d in self._declarative.definitions().values()
        ]

    def get_tool(self, name: str) -> ToolDefinition:
        if name.startswith("decl."):
            return self._declarative.tool(self._declarative.definitions()[name])
        if name not in self._catalog:
            raise KeyError(f"Unknown private tool: {name!r}")
        tool = copy.deepcopy(self._catalog[name])
        if name in {"http.request", "tools.define"} and tool.risk_level != "high":
            tool.risk_level = "high"
        if name == "fs.delete_file" and self._safety.delete_requires_human:
            tool.human_check = True
        return tool

    def authorize_call(
        self, context: NodeContext, tool: ToolDefinition, arguments: JsonDict
    ) -> None:
        """Engine hook: system lock precedes the single, persistent human gate."""
        path = self._validate_call(context, tool, arguments)
        if tool.name == "fs.delete_file" and not path.is_file():
            raise ValueError("fs.delete_file requires an existing regular file.")
        safety_context = None
        if tool.name == "fs.delete_file":
            info = path.stat()
            safety_context = {
                "resolved_path": str(path),
                "device": info.st_dev,
                "inode": info.st_ino,
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "ctime_ns": info.st_ctime_ns,
            }
        try:
            approval_tool = (
                replace(tool, human_check=True) if self._http_review(tool, arguments) else tool
            )
            require_tool_human_check(
                context, approval_tool, arguments, safety_context=safety_context
            )
        except HumanCheckRequired as pending:
            if tool.name == "fs.delete_file" and self._safety.delete_requires_human:
                emit_security_event(
                    context,
                    {
                        "kind": "delete_requires_human",
                        "tool": tool.name,
                        "path": str(path),
                        "request_id": pending.request_id,
                        "message": "File deletion paused. Human confirmation is required.",
                    },
                    self.on_security_event,
                )
            raise
        if tool.human_check or self._http_review(tool, arguments):
            self._authorized_calls[id(context)] = (
                context,
                copy.deepcopy(tool),
                copy.deepcopy(arguments),
                path,
            )

    @staticmethod
    def _http_review(tool, arguments):
        return tool.name == "http.request" and arguments.get("method") not in {"GET", "HEAD"}

    def _path(self, raw: str) -> Path:
        if not raw.strip() or "\x00" in raw or ":" in raw or Path(raw).anchor:
            raise ValueError("Tool paths must be nonempty workspace-relative paths.")
        path = (self.workspace_root / raw).resolve()
        if not path.is_relative_to(self.workspace_root):
            raise PermissionError("Tool path escapes the configured workspace.")
        # Compare case-insensitively: on macOS and Windows ".DGS-Tools" is the same folder.
        parts = path.relative_to(self.workspace_root).parts
        if parts and parts[0].casefold() == TEMPLATE_DIRECTORY:
            raise PermissionError("Template storage is managed by tools.define.")
        return path

    def _read_text(self, path: Path) -> str:
        with path.open("rb") as stream:
            data = stream.read(self._limits["max_read_bytes"] + 1)
        if len(data) > self._limits["max_read_bytes"]:
            raise ValueError("File exceeds max_read_bytes.")
        return data.decode("utf-8")

    def _validate_call(
        self, context: NodeContext, tool: ToolDefinition, arguments: JsonDict
    ) -> Path | None:
        if tool != self.get_tool(tool.name):
            raise RuntimeError("Tool definition changed after authorization; retry the call.")
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be an object.")
        size = len(json.dumps(arguments, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        if not tool.name.startswith("fs.") and size > self._limits["max_write_bytes"]:
            raise ValueError("Tool arguments exceed max_write_bytes.")
        if tool.name.startswith("decl."):
            return None
        required, optional = TOOL_CONTRACTS[tool.name]
        if set(arguments) - (set(required) | set(optional)) or set(required) - set(arguments):
            raise ValueError(f"Incorrect arguments for {tool.name!r}.")
        for name, value in arguments.items():
            expected = {**required, **optional}[name]
            if type(value) is not expected:
                raise TypeError(f"Argument {name!r} must be {expected.__name__}.")
        if tool.name == "http.request":
            self._http.validate(arguments)
        if tool.name == "tools.define":
            self._declarative.validate(arguments["definition"])
        if not tool.name.startswith("fs."):
            return None
        self._safety.check_system_path(
            context, tool, arguments["path"], self.workspace_root, self.on_security_event
        )
        return self._path(arguments["path"])

    def execute(self, context: NodeContext, tool: ToolDefinition, arguments: JsonDict):
        # Human approval is performed by the engine; recheck paths immediately
        # before I/O, including after a paused call resumes.
        path = self._validate_call(context, tool, arguments)
        if tool.human_check or self._http_review(tool, arguments):
            receipt = self._authorized_calls.pop(id(context), None)
            if (
                receipt is None
                or receipt[0] is not context
                or receipt[1:] != (tool, arguments, path)
            ):
                raise PermissionError("This call requires engine-dispatched human authorization.")
        if tool.name.startswith("fs."):
            result = self._execute_file(tool, arguments, path)
            context.record_activity(
                "file.accessed", {"operation": tool.name, "path": result["path"]}
            )
        elif tool.name.startswith("decl."):
            with self._template_lock:
                result = self._declarative.execute(context, tool, arguments)
        elif tool.name == "tools.define":
            with self._template_lock:
                result = self._declarative.define(
                    context, arguments["definition"], arguments.get("overwrite", False)
                )
        elif tool.name == "http.request":
            result = self._http.request(context, arguments, self._limits["max_read_bytes"])
        else:
            result = execute_basic(
                tool.name, arguments, self._limits["max_entries"], self._limits["max_read_bytes"]
            )
        return result

    def _execute_file(self, tool, arguments, path):
        relative = path.relative_to(self.workspace_root).as_posix()
        if tool.name == "fs.stat":
            return {
                "path": relative,
                "exists": path.exists(),
                "is_file": path.is_file(),
                "is_directory": path.is_dir(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        if tool.name == "fs.list_directory":
            entries = []
            with os.scandir(path) as iterator:
                for entry in iterator:
                    if len(entries) == self._limits["max_entries"]:
                        return {"path": relative, "entries": entries, "truncated": True}
                    entry_path = Path(entry.path)
                    is_link = entry.is_symlink() or (
                        hasattr(entry_path, "is_junction") and entry_path.is_junction()
                    )
                    kind = (
                        "link"
                        if is_link
                        else "directory"
                        if entry.is_dir(follow_symlinks=False)
                        else "file"
                    )
                    entries.append({"name": entry.name, "type": kind})
            return {"path": relative, "entries": entries, "truncated": False}
        if tool.name == "fs.read_text":
            return {"path": relative, "content": self._read_text(path)}
        if tool.name == "fs.read_json":

            def invalid_constant(value):
                raise ValueError(f"Invalid JSON constant: {value}")

            return {
                "path": relative,
                "data": json.loads(self._read_text(path), parse_constant=invalid_constant),
            }
        if tool.name == "fs.write_text":
            data = arguments["content"].encode("utf-8")
            if len(data) > self._limits["max_write_bytes"]:
                raise ValueError("Content exceeds max_write_bytes.")
            mode = "wb" if arguments.get("overwrite", False) else "xb"
            with path.open(mode) as stream:
                stream.write(data)
            return {"path": relative, "bytes_written": len(data)}
        if tool.name == "fs.make_directory":
            path.mkdir(parents=arguments.get("parents", False), exist_ok=True)
            return {"path": relative, "is_directory": True}
        if tool.name == "fs.delete_file":
            if not path.is_file():
                raise ValueError("fs.delete_file requires an existing regular file.")
            path.unlink()
            return {"path": relative, "deleted": True}
        raise KeyError(f"Unknown private tool: {tool.name!r}")
