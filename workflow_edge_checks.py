"""Runtime context and callable contract for user-defined edge entry checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from workflow_definitions import EdgeDefinition, WorkflowDefinition
from workflow_executions import Task
from workflow_types import JsonDict


@dataclass
class EdgeCheckContext:
    workflow: WorkflowDefinition
    task: Task
    edge: EdgeDefinition


EntryCheck = Callable[[EdgeCheckContext, JsonDict], bool]
