"""Portable join rules and a user-defined predicate contract."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from workflow_types import JsonDict

if TYPE_CHECKING:
    from workflow_definitions import NodeDefinition, WorkflowDefinition
    from workflow_executions import Task


def normalize_start_condition(raw: JsonDict, incoming: list[str]) -> JsonDict:
    if not isinstance(raw, dict):
        raise ValueError("start_condition must be a JSON object or null.")
    unknown = set(raw) - {
        "required_edges",
        "scope",
        "group_by",
        "generation_by",
        "check",
        "config",
        "accept_skipped",
    }
    if unknown:
        raise ValueError(f"Unknown start_condition fields: {sorted(unknown)}")
    required = raw.get("required_edges", incoming)
    if (
        not isinstance(required, list)
        or not required
        or any(not isinstance(x, str) or not x for x in required)
    ):
        raise ValueError("required_edges must be a nonempty list of edge IDs.")
    if len(set(required)) != len(required) or len(set(incoming)) != len(incoming):
        raise ValueError("Join edges must have unique IDs; give parallel edges explicit IDs.")
    if set(required) != set(incoming):
        raise ValueError("required_edges must identify every incoming edge exactly once.")
    scope = raw.get("scope", "task")
    if scope not in ("task", "group"):
        raise ValueError("Start-condition scope must be 'task' or 'group'.")
    accept_skipped = raw.get("accept_skipped", scope == "task")
    if type(accept_skipped) is not bool:
        raise ValueError("accept_skipped must be a Boolean.")
    for name in ("group_by", "generation_by"):
        pointer = raw.get(name)
        if pointer is not None and (not isinstance(pointer, str) or not pointer.startswith("/")):
            raise ValueError(f"{name} must be a JSON Pointer beginning with '/'.")
    if scope == "group" and raw.get("group_by") is None:
        raise ValueError("Group joins require group_by to isolate unrelated tasks.")
    check = raw.get("check")
    if check is not None and (not isinstance(check, str) or not check.strip()):
        raise ValueError("Start-condition check must be a function name or null.")
    if not isinstance(raw.get("config", {}), dict):
        raise ValueError("Start-condition config must be a JSON object.")
    return {
        "required_edges": list(required),
        "scope": scope,
        "group_by": raw.get("group_by"),
        "generation_by": raw.get("generation_by"),
        "check": check,
        "config": copy.deepcopy(raw.get("config", {})),
        "accept_skipped": accept_skipped,
    }


@dataclass
class StartConditionContext:
    workflow: WorkflowDefinition
    node: NodeDefinition
    tasks: list[Task]
    arrivals: list[JsonDict]
    data: JsonDict


StartConditionCheck = Callable[[StartConditionContext, JsonDict], bool]
