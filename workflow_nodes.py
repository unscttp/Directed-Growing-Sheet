"""Runtime view handed to node handlers, hooks and tool backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from workflow_definitions import NodeDefinition, WorkflowDefinition
from workflow_executions import Task, TaskExecution
from workflow_logging import audit_execution, record_event
from workflow_memory import drain_checkpoints
from workflow_prompts import build_node_messages
from workflow_settings import DEFAULTS
from workflow_timeouts import check_timeout
from workflow_tools import ToolDefinition
from workflow_types import JsonDict

if TYPE_CHECKING:
    from workflow_replay import ActionReplay


@dataclass
class NodeResult:
    """Result returned by a node handler.

    ``values`` holds logical node outputs; ``NodeDefinition.outputs`` decides
    where they are written in ``Task.data``.
    """

    values: JsonDict = field(default_factory=dict)
    status: str = "completed"  # completed / waiting / failed
    local: JsonDict = field(default_factory=dict)
    error: str | None = None


@dataclass
class NodeContext:
    """Runtime-only view of one node execution. It is derived, never persisted.

    ``runtime`` is the engine that created the context. A context without a
    runtime can still record activity, but cannot dispatch actions, build
    prompts or write model memory.
    """

    workflow: WorkflowDefinition
    node: NodeDefinition
    task: Task
    execution: TaskExecution
    inputs: JsonDict
    pre_run_data: JsonDict = field(default_factory=dict)
    runtime: Any = field(default=None, repr=False, compare=False)
    replay: ActionReplay | None = field(default=None, repr=False, compare=False)

    # -- engine services -------------------------------------------------

    def _require_runtime(self) -> Any:
        if self.runtime is None:
            raise RuntimeError("This context is not attached to a workflow engine.")
        return self.runtime

    @property
    def settings(self) -> JsonDict:
        """Current global settings (memory_settings.json), or the defaults."""
        if self.runtime is None:
            return dict(DEFAULTS)
        return self.runtime.settings()

    @property
    def allowance(self):
        """The node's current ``AllowanceSet``, or ``None`` when unrestricted."""
        return self._require_runtime().allowances.resolve(self.node.allowance)

    def allowance_snapshot(self) -> JsonDict | None:
        allowance = self.allowance
        return allowance.to_dict() if allowance is not None else None

    @property
    def memory_writer(self):
        """The model callback that writes active memory for this node type, if any."""
        if self.runtime is None:
            return None
        return self.runtime.memory_writers.get(self.node.type)

    def check_deadline(self) -> None:
        """Cooperative timeout check; raises when the node deadline has passed."""
        if self.runtime is not None:
            check_timeout(self)

    def drain_memory(self) -> None:
        """Ask the memory writer to turn queued objective checkpoints into notes."""
        drain_checkpoints(self, self.memory_writer)

    def record_decision_round(self) -> None:
        """Count one model decision for the round/token checkpoint settings."""
        if self.runtime is not None:
            self.runtime.record_react_round(self)

    # -- actions and tools -------------------------------------------------

    def act(self, action: str, arguments: JsonDict | None = None) -> Any:
        """Dispatch a registered action through the node's allowance checks."""
        return self._require_runtime().perform_action(
            self, action, {} if arguments is None else arguments
        )

    def use_tool(self, name: str, arguments: JsonDict | None = None) -> Any:
        """Call a tool through the same allowance, risk and approval checks."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name must be a nonempty string.")
        return self.act(f"tool:{name}", arguments)

    # -- prompts -----------------------------------------------------------

    @property
    def prompt_context(self) -> JsonDict:
        return self._require_runtime().node_prompt_context(self)

    def prompt_messages(self, history: list[JsonDict] | None = None) -> list[JsonDict]:
        return build_node_messages(self, history)

    # -- memory and audit --------------------------------------------------

    def checkpoint(self, name: str, details: JsonDict | None = None):
        """Invoke a checkpoint declared in the node's ``memory_checkpoints``."""
        if name not in self.node.memory_checkpoints:
            raise ValueError(f"Undeclared memory checkpoint: {name!r}")
        return self._require_runtime().checkpoint(self, name, details)

    def record_activity(self, kind: str, details: JsonDict) -> str:
        """Append an objective event for work the host performed."""
        return record_event(self.task, kind, details, self.execution)

    def _record_change(self, operation: str, details: JsonDict, apply):
        request_id = self.record_activity(operation + ".requested", details)
        try:
            with audit_execution(self.task, self.execution):
                result = apply()
        except Exception as exc:
            self.record_activity(
                operation + ".failed",
                {"request_id": request_id, "error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        self.record_activity(operation + ".completed", {"request_id": request_id, **details})
        return result

    def write_skill(self, adapter, name: str, request: JsonDict):
        """Trusted host entry point for a ``SkillWriteInterface`` adapter."""
        return self._record_change(
            "skill.write", {"skill": name}, lambda: adapter.write_skill(self, name, request)
        )

    def add_tool(self, adapter, description, implementation_request: JsonDict):
        """Audit a host-approved request through a ``ToolCreationInterface`` adapter."""
        details = {"requested_tool": description.to_dict()}

        def create():
            tool = adapter.add_tool(self, description, implementation_request)
            if not isinstance(tool, ToolDefinition):
                raise TypeError("Tool creation adapter must return a ToolDefinition.")
            details["created_tool"] = tool.to_dict()
            return tool

        return self._record_change("tool.creation", details, create)
