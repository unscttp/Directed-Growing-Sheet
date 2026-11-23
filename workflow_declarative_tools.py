"""Declarative tool recipes: typed references and resumable, audited tool steps."""

import copy
import hashlib
import json
import re

from workflow_approvals import HumanCheckRequired
from workflow_basic_tools import TOOL_CONTRACTS
from workflow_declarative_checks import check_tree_bounds, expression_type
from workflow_memory_store import atomic_write
from workflow_schema import validate_document
from workflow_tools import ToolDefinition
from workflow_utils import get_value

TYPES = {"string": str, "object": dict, "array": list, "boolean": bool, "integer": int}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def resolve(value, inputs, results):
    if isinstance(value, dict):
        if "$input" in value:
            return copy.deepcopy(inputs[value["$input"]])
        if "$step" in value:
            return copy.deepcopy(get_value(results[value["$step"]], value.get("path", "")))
        return {k: resolve(v, inputs, results) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, inputs, results) for v in value]
    return value


class DeclarativeTools:
    def __init__(self, module):
        self.module = module

    def store_path(self):
        directory = self.module.workspace_root / ".dgs-tools"
        if directory.is_symlink() or (
            hasattr(directory, "is_junction") and directory.is_junction()
        ):
            raise PermissionError("Template directory cannot be a link.")
        path = directory / "definitions.json"
        if path.is_symlink():
            raise PermissionError("Template store cannot be a link.")
        return path

    def definitions(self):
        path = self.store_path()
        if not path.exists():
            return {}
        if path.stat().st_size > self.module._limits["max_write_bytes"]:
            raise ValueError("Template store exceeds size limit.")
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Invalid template store.")
        for name, definition in values.items():
            self.validate(definition)
            if name != definition["name"]:
                raise ValueError("Template name mismatch.")
        return values

    def validate(self, definition):
        check_tree_bounds(definition)
        validate_document(definition, "declarative_tool")
        if not definition["description"].strip():
            raise ValueError("Description must be nonempty.")
        params = {name: TYPES[kind] for name, kind in definition["parameters"].items()}
        steps = definition["steps"]
        previous = {}

        for step in steps:
            if (
                not isinstance(step, dict)
                or set(step) != {"id", "tool", "arguments"}
                or not isinstance(step["id"], str)
                or not re.fullmatch(r"[a-z][a-z0-9_]*", step["id"])
                or step["id"] in previous
                or not isinstance(step["tool"], str)
                or step["tool"] not in self.module._declarative_allowed_tools
                or step["tool"] == "tools.define"
                or not isinstance(step["arguments"], dict)
            ):
                raise ValueError("Invalid template step.")
            args = step["arguments"]
            required, optional = TOOL_CONTRACTS[step["tool"]]
            if set(required) - args.keys() or args.keys() - (required.keys() | optional.keys()):
                raise ValueError("Template step has incorrect arguments.")
            for name, value in args.items():
                inferred = expression_type(value, params, previous)
                expected = {**required, **optional}[name]
                if inferred is not None and inferred is not expected:
                    raise ValueError("Template argument has incorrect static type.")
            if step["tool"] == "http.request":
                # Keep destination and method fixed; parameters belong in body and headers.
                if type(args["url"]) is not str or type(args["method"]) is not str:
                    raise ValueError("HTTP template URL and method must be literal.")
                self.module._http.validate({"url": args["url"], "method": args["method"]})
                query = args.get("query", {})
                if not any(key in query for key in ("$input", "$step")):
                    for value in query.values():
                        inferred = expression_type(value, params, previous)
                        if inferred is not None and inferred not in (str, int, bool):
                            raise ValueError("HTTP query expression must produce a scalar.")
                headers = args.get("headers", {})
                if not any(key in headers for key in ("$input", "$step")):
                    placeholders = {}
                    for header, value in headers.items():
                        inferred = expression_type(value, params, previous)
                        if inferred is not None and inferred is not str:
                            raise ValueError("HTTP header expression must produce a string.")
                        placeholders[header] = value if isinstance(value, str) else "placeholder"
                    self.module._http.validate(
                        {
                            **args,
                            "headers": placeholders,
                            "query": {},
                            "body": args.get("body", "")
                            if isinstance(args.get("body", ""), str)
                            else "",
                        }
                    )
            previous[step["id"]] = step["tool"]
        return definition

    def check_static(self, context, definition):
        try:
            self.validate(definition)
            allowance = context.allowance
            required = []
            for step in definition["steps"]:
                tool = self.module.get_tool(step["tool"])
                action = "tool:" + tool.name
                if allowance is not None and (
                    not allowance.allows(action) or not allowance.allows_tool_risk(tool.risk_level)
                ):
                    raise PermissionError(
                        f"Template permission exceeds current allowance: {action}"
                    )
                if action not in required:
                    required.append(action)
                path = step["arguments"].get("path")
                if tool.name.startswith("fs.") and isinstance(path, str):
                    self.module._path(path)
                    self.module._safety.check_system_path(
                        context,
                        tool,
                        path,
                        self.module.workspace_root,
                        self.module.on_security_event,
                    )
            context.record_activity(
                "tool.template.checked",
                {
                    "name": definition["name"],
                    "sha256": digest(definition),
                    "required_actions": required,
                },
            )
        except Exception as exc:
            context.record_activity(
                "tool.template.rejected", {"error_type": type(exc).__name__, "error": str(exc)}
            )
            raise

    def tool(self, definition):
        return ToolDefinition(
            definition["name"],
            "high",
            description=definition["description"],
            config={"definition_sha256": digest(definition)},
            input_schema={
                "type": "object",
                "properties": {k: {"type": v} for k, v in definition["parameters"].items()},
                "required": list(definition["parameters"]),
                "additionalProperties": False,
            },
        )

    def define(self, context, definition, overwrite=False):
        self.check_static(context, definition)
        values = self.definitions()
        exists = definition["name"] in values
        if exists and not overwrite:
            raise FileExistsError("Template exists; set overwrite=true to edit.")
        values[definition["name"]] = copy.deepcopy(definition)
        data = json.dumps(values, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")
        if len(data) > self.module._limits["max_write_bytes"]:
            raise ValueError("Template store exceeds max_write_bytes.")
        path = self.store_path()
        self.module._safety.check_system_path(
            context,
            ToolDefinition("fs.write_text", "medium"),
            ".dgs-tools/definitions.json",
            self.module.workspace_root,
            self.module.on_security_event,
        )
        path.parent.mkdir(exist_ok=True)
        atomic_write(path, data)
        tool = self.tool(definition)
        context.record_activity(
            "tool.updated" if exists else "tool.added",
            {"tool": tool.to_dict(), "definition": definition},
        )
        return {"tool": tool.to_dict()}

    def execute(self, context, tool, args):
        definition = self.definitions()[tool.name]
        if self.tool(definition) != tool:
            raise RuntimeError("Template changed; retry with its new definition.")
        self.check_static(context, definition)
        params = definition["parameters"]
        if set(args) != set(params) or any(
            type(args[k]) is not TYPES[t] for k, t in params.items()
        ):
            raise ValueError("Template arguments do not match declared parameters.")
        key = digest({"tool": tool.name, "arguments": args})
        runs = context.execution.memory_state.setdefault("declarative_runs", {})
        state = runs.setdefault(key, {"definition": digest(definition), "results": {}})
        if state["definition"] != digest(definition):
            raise RuntimeError("Template changed while execution was pending.")
        if "error" in state:
            raise RuntimeError(
                "Previous step failed; host must inspect effects before clearing this run."
            )
        for step in definition["steps"]:
            if step["id"] in state["results"]:
                continue
            try:
                arguments = resolve(step["arguments"], args, state["results"])
                result = context.use_tool(step["tool"], arguments)
            except HumanCheckRequired:
                raise
            except Exception as exc:
                state["error"] = str(exc)
                raise
            state["results"][step["id"]] = copy.deepcopy(result)
        result = copy.deepcopy(state["results"])
        del runs[key]
        return {"steps": result, "result": result[definition["steps"][-1]["id"]]}
