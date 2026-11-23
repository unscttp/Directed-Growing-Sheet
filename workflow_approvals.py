"""Persistent human-review requests; the host application supplies the human UI."""

from __future__ import annotations

import copy
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from workflow_logging import record_event
from workflow_types import JsonDict
from workflow_utils import new_id

if TYPE_CHECKING:
    from workflow_executions import Task
    from workflow_nodes import NodeContext


_approval_server = ContextVar("approval_server", default=None)


@contextmanager
def approval_server(server):
    token = _approval_server.set(server)
    try:
        yield
    finally:
        _approval_server.reset(token)


class HumanCheckRequired(Exception):
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(f"Human check required: {request_id}")


def require_human_check(context: NodeContext, key: str, details: JsonDict) -> None:
    """Consume one matching approval or pause before the protected operation."""
    server = _approval_server.get()
    if server is not None:
        key = f"mcp:{server}:{key}"
        details = {**details, "mcp_server": server}
    payload = json.loads(json.dumps(details, sort_keys=True, allow_nan=False))
    checks = context.execution.human_checks
    for request in reversed(checks):
        if request["key"] != key or request["status"] not in {"pending", "approved", "rejected"}:
            continue
        if json.dumps(request["details"], sort_keys=True, allow_nan=False) != json.dumps(
            payload, sort_keys=True
        ):
            request["status"] = "superseded"
            record_event(
                context.task,
                "approval.superseded",
                {"request_id": request["id"]},
                context.execution,
            )
            continue
        if request["status"] == "rejected":
            raise PermissionError(f"Human rejected request {request['id']!r}.")
        if request["status"] == "approved":
            request["status"] = "consumed"
            context.task.revision += 1
            record_event(
                context.task, "approval.consumed", {"request_id": request["id"]}, context.execution
            )
            return
        raise HumanCheckRequired(request["id"])
    request = {
        "id": new_id("approval"),
        "execution_id": context.execution.id,
        "node_id": context.node.id,
        "key": key,
        "details": payload,
        "status": "pending",
        "reviewer": None,
        "comment": "",
    }
    checks.append(request)
    record_event(context.task, "approval.requested", request, context.execution)
    context.task.revision += 1
    raise HumanCheckRequired(request["id"])


def pending_human_checks(task: Task) -> list[JsonDict]:
    return [
        copy.deepcopy(request)
        for execution in task.executions
        for request in execution.human_checks
        if request["status"] == "pending"
    ]


def resolve_human_check(
    task: Task, request_id: str, approved: bool, *, reviewer: str, comment: str = ""
) -> None:
    """Call only from an authenticated human-facing application, never an agent tool."""
    if type(approved) is not bool:
        raise TypeError("Human decision must be a Boolean.")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("A reviewer identity is required.")
    if not isinstance(comment, str):
        raise TypeError("Review comment must be a string.")
    for execution in task.executions:
        for request in execution.human_checks:
            if request["id"] != request_id:
                continue
            if request["status"] != "pending" or execution.status != "waiting":
                raise ValueError("Request is not awaiting a human decision.")
            request.update(
                status="approved" if approved else "rejected", reviewer=reviewer, comment=comment
            )
            # A request marked "on_reject": "continue" lets the node go on when declined;
            # the protected operation sees the rejection and skips itself.
            proceed = approved or request["details"].get("on_reject") == "continue"
            execution.status = "ready" if proceed else "failed"
            if approved and request["details"].get("kind") == "timeout":
                timing = execution.timing
                timing["deadline"] = time.time() + timing["interval"]
                previous = timing.pop("resume_status", "ready")
                execution.status = "waiting_start" if previous == "waiting_start" else "ready"
                if any(r["status"] == "pending" for r in execution.human_checks):
                    execution.status = "waiting"
                if execution.routing_state.get("status") == "pending":
                    execution.status = "waiting"
                request["status"] = "consumed"
                record_event(
                    task, "node.timeout_extended", {"deadline": timing["deadline"]}, execution
                )
            execution.error = None if proceed else f"Human rejected request {request_id!r}."
            task.revision += 1
            record_event(
                task,
                "approval.resolved",
                {
                    "request_id": request_id,
                    "approved": approved,
                    "reviewer": reviewer,
                    "comment": comment,
                },
                execution,
            )
            if not approved:
                task.settle_status("human rejected a request")
            return
    raise KeyError(f"Unknown human-check request: {request_id!r}")
