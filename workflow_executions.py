from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path

from workflow_definitions import WorkflowDefinition
from workflow_logging import record_event
from workflow_memory import empty_active_memory, restore_active_memory
from workflow_memory_store import load_snapshot, save_snapshot
from workflow_schema import validate_document
from workflow_timeouts import arm_timeout
from workflow_types import JsonDict
from workflow_utils import new_id

ACTIVE_EXECUTION_STATUSES = frozenset({"ready", "running", "waiting", "waiting_start"})
TERMINAL_TASK_STATUSES = frozenset({"completed", "failed"})


@dataclass
class TaskExecution:
    """
    One execution of one node.

    Multiple active TaskExecution objects inside the same Task provide the
    foundation for parallel execution.
    """

    id: str
    node_id: str

    # ready / running / waiting / waiting_start / completed / failed / skipped / joined
    status: str = "ready"
    attempt: int = 0

    input_data: JsonDict = field(default_factory=dict)
    output_data: JsonDict = field(default_factory=dict)
    local: JsonDict = field(default_factory=dict)
    error: str | None = None
    human_checks: list[JsonDict] = field(default_factory=list)
    continuation: JsonDict = field(default_factory=dict)
    arrival: JsonDict | None = None
    joined_from: list[JsonDict] = field(default_factory=list)
    scheduled: bool = False
    memory_state: JsonDict = field(default_factory=dict)
    routing_state: JsonDict = field(default_factory=dict)
    timing: JsonDict = field(default_factory=dict)
    skip_path: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, node_id: str) -> TaskExecution:
        return cls(id=new_id("exec"), node_id=node_id)

    @classmethod
    def from_dict(cls, raw: JsonDict) -> TaskExecution:

        validate_document(raw, "execution")
        return cls(
            id=raw["id"],
            node_id=raw["node_id"],
            status=raw.get("status", "ready"),
            attempt=int(raw.get("attempt", 0)),
            input_data=copy.deepcopy(raw.get("input_data", {})),
            output_data=copy.deepcopy(raw.get("output_data", {})),
            local=copy.deepcopy(raw.get("local", {})),
            error=raw.get("error"),
            human_checks=copy.deepcopy(raw.get("human_checks", [])),
            continuation=copy.deepcopy(raw.get("continuation", {})),
            arrival=copy.deepcopy(raw.get("arrival")),
            joined_from=copy.deepcopy(raw.get("joined_from", [])),
            scheduled=raw.get("scheduled", False),
            memory_state=copy.deepcopy(raw.get("memory_state", {})),
            routing_state=copy.deepcopy(raw.get("routing_state", {})),
            timing=copy.deepcopy(raw.get("timing", {})),
            skip_path=copy.deepcopy(raw.get("skip_path", [])),
        )


@dataclass
class Task:
    """
    Persistent workflow instance.

    There is intentionally no separate State object:
    - business data: Task.data
    - execution state: Task.executions / Task.status
    - node-local state: TaskExecution.local
    """

    id: str
    workflow_id: str
    workflow_version: str

    data: JsonDict = field(default_factory=dict)
    executions: list[TaskExecution] = field(default_factory=list)

    status: str = "running"
    revision: int = 0
    metadata: JsonDict = field(default_factory=dict)
    events: list[JsonDict] = field(default_factory=list)
    active_memory: JsonDict = field(default_factory=empty_active_memory)

    def __post_init__(self) -> None:
        if not self.events:
            record_event(
                self,
                "task.created",
                {
                    "workflow_id": self.workflow_id,
                    "data": self.data,
                    "start_nodes": [e.node_id for e in self.executions],
                },
            )

    @classmethod
    def create(
        cls,
        workflow: WorkflowDefinition,
        data: JsonDict | None = None,
    ) -> Task:
        task = cls(
            id=new_id("task"),
            workflow_id=workflow.id,
            workflow_version=workflow.format_version,
            data=copy.deepcopy(data or {}),
            executions=[TaskExecution.create(node_id) for node_id in workflow.start],
        )
        for execution in task.executions:
            arm_timeout(workflow.node(execution.node_id), execution)
        return task

    def active_executions(self) -> list[TaskExecution]:
        return [
            execution
            for execution in self.executions
            if execution.status in ACTIVE_EXECUTION_STATUSES
        ]

    def set_status(self, status: str, reason: str) -> None:
        """Change the task status and record the transition."""
        if status == self.status:
            return
        before, self.status = self.status, status
        record_event(self, "task.status_changed", {"from": before, "to": status, "reason": reason})
        if status in TERMINAL_TASK_STATUSES:
            record_event(self, f"task.{status}", {"reason": reason})

    def settle_status(self, reason: str) -> None:
        """Mark the task completed or failed once no execution can make progress."""
        if self.status == "merged" or self.active_executions():
            return
        failed = any(execution.status == "failed" for execution in self.executions)
        self.set_status("failed" if failed else "completed", reason)

    def completed_nodes(self) -> set[str]:
        return {
            execution.node_id for execution in self.executions if execution.status == "completed"
        }

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "workflow_version": self.workflow_version,
            "status": self.status,
            "revision": self.revision,
            "data": copy.deepcopy(self.data),
            "executions": [asdict(execution) for execution in self.executions],
            "metadata": copy.deepcopy(self.metadata),
            "events": copy.deepcopy(self.events),
            "active_memory": copy.deepcopy(self.active_memory),
        }

    @classmethod
    def from_dict(cls, raw: JsonDict) -> Task:

        validate_document(raw, "task")
        if "memory_streams" in raw:
            raise ValueError("Use Task.load(path) to resolve JSONL memory streams.")
        task = cls(
            id=raw["id"],
            workflow_id=raw["workflow_id"],
            workflow_version=raw.get("workflow_version", "1.0"),
            status=raw.get("status", "running"),
            revision=int(raw.get("revision", 0)),
            data=copy.deepcopy(raw.get("data", {})),
            executions=[TaskExecution.from_dict(item) for item in raw.get("executions", [])],
            metadata=copy.deepcopy(raw.get("metadata", {})),
            events=copy.deepcopy(raw.get("events", [])),
            active_memory=restore_active_memory(raw.get("active_memory", [])),
        )
        if not raw.get("events"):
            task.events[0]["type"] = "task.restored_legacy"
            task.events[0]["details"]["prior_history_available"] = False
        # Older branch decisions and manually appended notes could have no citations.
        appended = {
            e["details"]["memory_id"]: e["id"]
            for e in task.events
            if e["type"] == "memory.appended"
            and isinstance(e["details"], dict)
            and "memory_id" in e["details"]
        }
        for section in task.active_memory.values():
            for note in section:
                if not note.get("references"):
                    event_id = appended.get(note.get("id")) or record_event(
                        task, "memory.legacy_imported", {"memory_id": note.get("id")}
                    )
                    note["references"] = [event_id]
        return task

    def save(self, path: str | Path) -> None:
        save_snapshot(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> Task:
        return cls.from_dict(load_snapshot(path))
