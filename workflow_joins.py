"""Serializable edge-arrival queues and synchronous task/branch joins."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from workflow_definitions import WorkflowDefinition
from workflow_executions import Task, TaskExecution
from workflow_logging import record_event
from workflow_memory_store import load_snapshot, save_snapshot
from workflow_start_conditions import StartConditionCheck, StartConditionContext
from workflow_types import JsonDict
from workflow_utils import get_value, new_id


class JoinConflict(ValueError):
    pass


def merge_task_data(items: list[JsonDict]) -> JsonDict:
    """Merge objects recursively; differing values or types fail with their path."""

    def merge(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            result = copy.deepcopy(left)
            for key, value in right.items():
                pointer = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
                result[key] = (
                    merge(result[key], value, pointer) if key in result else copy.deepcopy(value)
                )
            return result
        if json.dumps(left, sort_keys=True, allow_nan=False) != json.dumps(
            right, sort_keys=True, allow_nan=False
        ):
            raise JoinConflict(f"Conflicting task data at {path or '/'}.")
        return copy.deepcopy(left)

    result: JsonDict = {}
    for item in items:
        result = merge(result, item, "")
    return result


class TaskPool:
    """Own all participating tasks and persist them together for cross-task joins.

    Use from one scheduler thread. Distributed callers must provide locking and
    atomic persistence around scheduling; this is not a distributed queue.
    """

    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self.next_arrival = 0

    def add(self, task: Task) -> None:
        # Replaces an earlier in-memory copy when the host restores a saved task.
        self.tasks[task.id] = task
        for execution in task.executions:
            if execution.arrival:
                self.next_arrival = max(
                    self.next_arrival, execution.arrival.get("sequence", -1) + 1
                )

    def ready(self) -> list[tuple[Task, TaskExecution]]:
        return [
            (task, execution)
            for task in self.tasks.values()
            for execution in task.executions
            if execution.status == "ready"
        ]

    def to_dict(self) -> JsonDict:
        return {
            "tasks": [task.to_dict() for task in self.tasks.values()],
            "next_arrival": self.next_arrival,
        }

    @classmethod
    def from_dict(cls, raw: JsonDict) -> TaskPool:
        pool = cls()
        for item in raw.get("tasks", []):
            if item["id"] in pool.tasks:
                raise ValueError("Task pool contains duplicate task IDs.")
            pool.add(Task.from_dict(item))
        pool.next_arrival = max(pool.next_arrival, raw.get("next_arrival", 0))
        return pool

    def save(self, path: str | Path) -> None:
        save_snapshot(path, self.to_dict(), pool=True)

    @classmethod
    def load(cls, path: str | Path) -> TaskPool:
        return cls.from_dict(load_snapshot(path, pool=True))

    def arrival_record(self, workflow, task, execution, edge) -> JsonDict:
        rule = workflow.start_rule(edge.to_node)
        group = task.id
        generation = None
        if rule:
            if rule["scope"] == "group":
                group = get_value(task.data, rule["group_by"])
                if group is None:
                    raise ValueError("A group join requires a non-null correlation value.")
            if rule["generation_by"]:
                generation = get_value(task.data, rule["generation_by"])
                if generation is None:
                    raise ValueError("Join generation must not be null.")
        record = {
            "edge_id": workflow.edge_key(edge),
            "source_execution_id": execution.id,
            "source_task_id": task.id,
            "group": copy.deepcopy(group),
            "generation": copy.deepcopy(generation),
            "rule": rule,
            "data": copy.deepcopy(task.data),
            "resolution": "entered",
        }
        json.dumps(record, allow_nan=False)
        return record

    def stamp(self, record: JsonDict) -> None:
        record["sequence"] = self.next_arrival
        self.next_arrival += 1

    def refresh(
        self, workflow: WorkflowDefinition, checks: dict[str, StartConditionCheck]
    ) -> list[tuple[Task, TaskExecution]]:
        """Release every join round whose required arrivals are all present.

        Arrivals are grouped by (target node, correlation group, generation)
        and consumed FIFO per edge, one arrival per required edge per round.
        """
        released: list[tuple[Task, TaskExecution]] = []
        for group in self._waiting_groups(workflow).values():
            group.sort(key=lambda pair: pair[1].arrival["sequence"])
            node = workflow.node(group[0][1].node_id)
            rule = workflow.start_rule(node.id)
            if rule is None or any(item.arrival["rule"] != rule for _, item in group):
                self._fail(
                    group, "Start condition changed after arrival; resubmit under the new workflow."
                )
                continue
            while True:
                selected = self._next_round(group, rule)
                if selected is None:
                    break
                try:
                    data = self._combined_data(selected)
                    if not self._check_passes(workflow, node, rule, checks, selected, data):
                        break
                except Exception as exc:
                    self._fail(selected, f"{type(exc).__name__}: {exc}")
                    continue
                released.append(self._release(workflow, node, selected, data))
        return released

    def _waiting_groups(self, workflow):
        groups: dict[str, list[tuple[Task, TaskExecution]]] = {}
        for task in list(self.tasks.values()):
            if (task.workflow_id, task.workflow_version) != (workflow.id, workflow.format_version):
                continue
            for execution in task.executions:
                if execution.status != "waiting_start" or execution.arrival is None:
                    continue
                arrival = execution.arrival
                key = json.dumps(
                    [execution.node_id, arrival["group"], arrival["generation"]], sort_keys=True
                )
                groups.setdefault(key, []).append((task, execution))
        return groups

    @staticmethod
    def _next_round(group, rule):
        """The oldest waiting arrival for each required edge, or None if one is missing."""
        selected = []
        for edge_id in rule["required_edges"]:
            found = next(
                (
                    pair
                    for pair in group
                    if pair[1].status == "waiting_start" and pair[1].arrival["edge_id"] == edge_id
                ),
                None,
            )
            if found is None:
                return None
            selected.append(found)
        return selected

    @staticmethod
    def _active_receipts(selected):
        receipts = [copy.deepcopy(execution.arrival) for _, execution in selected]
        return receipts, [r for r in receipts if r.get("resolution") != "skipped"]

    def _combined_data(self, selected) -> JsonDict:
        sources = {task.id: task for task, _ in selected}
        if len(sources) == 1:
            return copy.deepcopy(next(iter(sources.values())).data)
        # Each source task contributes the snapshot of its latest real arrival.
        _, active = self._active_receipts(selected)
        snapshots = {}
        for receipt in sorted(active, key=lambda r: r["sequence"]):
            snapshots[receipt["source_task_id"]] = receipt["data"]
        return merge_task_data(list(snapshots.values()))

    def _check_passes(self, workflow, node, rule, checks, selected, data) -> bool:
        receipts, active = self._active_receipts(selected)
        if rule["check"] is None or not active:
            return True
        if rule["check"] not in checks:
            raise KeyError(f"Unknown node start check: {rule['check']!r}")
        sources = list({task.id: task for task, _ in selected}.values())
        allowed = checks[rule["check"]](
            StartConditionContext(workflow, node, sources, receipts, copy.deepcopy(data)),
            copy.deepcopy(rule["config"]),
        )
        if type(allowed) is not bool:
            raise TypeError("Node start check must return a Boolean.")
        for source in sources:
            record_event(
                source,
                "join.checked",
                {"node_id": node.id, "check": rule["check"], "allowed": allowed},
            )
        return allowed

    def _release(self, workflow, node, selected, data):
        sources = {task.id: task for task, _ in selected}
        receipts, active = self._active_receipts(selected)
        joined = TaskExecution.create(node.id)
        joined.joined_from = receipts
        deadlines = [e.timing for _, e in selected if "deadline" in e.timing]
        if deadlines:
            joined.timing = copy.deepcopy(min(deadlines, key=lambda timing: timing["deadline"]))
            joined.timing.pop("resume_status", None)
        if not active:
            joined.status = "skipped"
            joined.skip_path = list(dict.fromkeys(key for _, e in selected for key in e.skip_path))

        if len(sources) == 1:
            owner = next(iter(sources.values()))
            owner.executions.append(joined)
            owner.set_status("running", "join released")
        else:
            owner = self._merged_task(workflow, sources, receipts, joined, data)
        record_event(
            owner,
            "join.released",
            {"source_task_ids": list(sources), "arrivals": receipts},
            joined,
        )
        for task, execution in selected:
            execution.status = "joined"
            execution.local["joined_into"] = {"task_id": owner.id, "execution_id": joined.id}
            task.revision += 1
            record_event(
                task,
                "join.consumed",
                {"owner_task_id": owner.id, "owner_execution_id": joined.id},
                execution,
            )
        for task in sources.values():
            if task is not owner and not task.active_executions():
                failed = any(e.status == "failed" for e in task.executions)
                task.set_status("failed" if failed else "merged", "joined into another task")
            elif task is owner:
                task.settle_status("join released")
        return owner, joined

    def _merged_task(self, workflow, sources, receipts, joined, data) -> Task:
        """A new task continuing at the join node, carrying all source memory."""
        owner = Task(
            id=new_id("task"),
            workflow_id=workflow.id,
            workflow_version=workflow.format_version,
            data=data,
            executions=[joined],
            metadata={
                "joined_task_ids": list(sources),
                "join_group": receipts[0]["group"],
                "join_generation": receipts[0]["generation"],
            },
        )
        self.add(owner)
        # Copy each source event once, keeping original IDs, task IDs, timestamps
        # and source-local sequence numbers for provenance.
        seen = {event["id"] for event in owner.events}
        for source in sources.values():
            for event in source.events:
                if event["id"] not in seen:
                    owner.events.append(copy.deepcopy(event))
                    seen.add(event["id"])
        # Archive first, so a note archived in one source cannot reappear as current.
        memory_ids = set()
        for section in ("archive", "current"):
            for source in sources.values():
                for note in source.active_memory[section]:
                    if note["id"] not in memory_ids:
                        owner.active_memory[section].append(copy.deepcopy(note))
                        memory_ids.add(note["id"])
        return owner

    @staticmethod
    def _fail(items, error) -> None:
        for task, execution in items:
            execution.status = "failed"
            execution.error = error
            task.revision += 1
            record_event(task, "join.failed", {"error": error}, execution)
            task.settle_status("join failed")
