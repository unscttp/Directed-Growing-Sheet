"""Replaceable tool execution backend and a standard-library local implementation."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from workflow_approvals import require_human_check
from workflow_logging import record_registration
from workflow_risk import risk_rank
from workflow_schema import validate_document, validate_schema
from workflow_types import JsonDict

if TYPE_CHECKING:
    from workflow_nodes import NodeContext


@dataclass
class ToolDefinition:
    name: str
    risk_level: str
    description: str = ""
    config: JsonDict = field(default_factory=dict)
    human_check: bool = False
    input_schema: JsonDict = field(default_factory=lambda: {"type": "object"})

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool name must be a nonempty string.")
        risk_rank(self.risk_level)
        if type(self.human_check) is not bool:
            raise ValueError("Tool human_check must be a Boolean.")
        if not isinstance(self.config, dict):
            raise ValueError("Tool config must be a JSON object.")
        json.dumps(self.config, allow_nan=False)
        if not isinstance(self.input_schema, dict) or self.input_schema.get("type") != "object":
            raise ValueError("Tool input_schema must describe a JSON object.")
        json.dumps(self.input_schema, allow_nan=False)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: JsonDict) -> ToolDefinition:

        validate_document(raw, "tool")
        return cls(
            name=raw["name"],
            risk_level=raw["risk_level"],
            description=raw.get("description", ""),
            config=copy.deepcopy(raw.get("config", {})),
            human_check=raw.get("human_check", False),
            input_schema=copy.deepcopy(raw.get("input_schema", {"type": "object"})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> ToolDefinition:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


ToolHandler = Callable[["NodeContext", ToolDefinition, JsonDict], Any]


def require_tool_human_check(
    context: NodeContext,
    tool: ToolDefinition,
    arguments: JsonDict,
    *,
    safety_context: JsonDict | None = None,
) -> None:
    if tool.human_check:
        details = {
            "kind": "tool",
            "tool": tool.to_dict(),
            "arguments": arguments,
            "allowance": context.node.allowance,
        }
        if safety_context is not None:
            details["safety_context"] = safety_context
        require_human_check(context, f"tool:{tool.name}", details)


class ToolModule(Protocol):
    """Describe before execution so the engine can enforce risk allowances.

    get_tool must be side-effect free. execute must execute the described tool
    and return a JSON-compatible value. Backends are trusted application code.
    """

    def get_tool(self, name: str) -> ToolDefinition: ...

    def execute(self, context: NodeContext, tool: ToolDefinition, arguments: JsonDict) -> Any: ...


class LocalToolModule:
    """Default backend; replace it with any implementation of ToolModule."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDefinition, ToolHandler]] = {}

    def register(self, tool: ToolDefinition, handler: ToolHandler) -> None:
        if not callable(handler):
            raise TypeError("Tool handler must be callable.")
        validated = ToolDefinition.from_dict(tool.to_dict())
        added = validated.name not in self._tools
        self._tools[validated.name] = (validated, handler)
        if added:
            record_registration(validated)

    def get_tool(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}")
        return copy.deepcopy(self._tools[name][0])

    def list_tools(self) -> list[ToolDefinition]:
        return [copy.deepcopy(tool) for tool, _ in self._tools.values()]

    def set_human_check(self, name: str, required: bool = True) -> None:
        """Require human review before each engine-dispatched invocation."""
        tool = replace(self.get_tool(name), human_check=required)
        self.register(tool, self._tools[name][1])

    def execute(self, context: NodeContext, tool: ToolDefinition, arguments: JsonDict) -> Any:
        """Check the arguments against the tool's ``input_schema``, then run the handler."""
        current = self.get_tool(tool.name)
        if current != tool:
            raise RuntimeError("Tool definition changed after authorization; retry the call.")
        validate_schema(arguments, tool.input_schema, path="arguments")
        return self._tools[tool.name][1](context, tool, arguments)


class CompositeToolModule:
    """Serve several local tool backends as one, e.g. built-in tools plus skills.

    Each tool name must be unique across the members. Calls are routed to the
    member that owns the tool, including its ``authorize_call`` safety hook.
    """

    def __init__(self, *modules) -> None:
        if not modules:
            raise ValueError("CompositeToolModule needs at least one tool module.")
        for module in modules:
            if not callable(getattr(module, "get_tool", None)) or not callable(
                getattr(module, "execute", None)
            ):
                raise TypeError("Every member must implement get_tool and execute.")
        self.modules = list(modules)
        self.list_tools()  # reject duplicate names early

    def list_tools(self) -> list[ToolDefinition]:
        tools: list[ToolDefinition] = []
        owners: dict[str, int] = {}
        for index, module in enumerate(self.modules):
            discover = getattr(module, "list_tools", None)
            for tool in discover() if callable(discover) else []:
                if tool.name in owners:
                    raise ValueError(f"Tool {tool.name!r} is provided by two member modules.")
                owners[tool.name] = index
                tools.append(tool)
        return tools

    def _owner(self, name: str):
        found = []
        for module in self.modules:
            try:
                module.get_tool(name)
            except KeyError:
                continue
            found.append(module)
        if not found:
            raise KeyError(f"Unknown tool: {name!r}")
        if len(found) > 1:
            raise ValueError(f"Tool {name!r} is provided by two member modules.")
        return found[0]

    def get_tool(self, name: str) -> ToolDefinition:
        return self._owner(name).get_tool(name)

    def authorize_call(self, context: NodeContext, tool: ToolDefinition, arguments: JsonDict):
        authorize = getattr(self._owner(tool.name), "authorize_call", None)
        if authorize is not None:
            return authorize(context, tool, arguments)
        return require_tool_human_check(context, tool, arguments)

    def execute(self, context: NodeContext, tool: ToolDefinition, arguments: JsonDict) -> Any:
        return self._owner(tool.name).execute(context, tool, arguments)

    @property
    def skill_packages(self) -> dict:
        packages: dict = {}
        for module in self.modules:
            packages.update(getattr(module, "skill_packages", {}))
        return packages

    @property
    def safety_policy(self):
        for module in self.modules:
            policy = getattr(module, "safety_policy", None)
            if policy is not None:
                return policy
        return None
