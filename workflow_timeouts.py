"""Cooperative persistent deadlines, checked by the engine and scheduler."""

import time

from workflow_approvals import require_human_check
from workflow_logging import record_event


def arm_timeout(node, execution, now=None):
    if node.timeout_seconds is not None and "deadline" not in execution.timing:
        now = time.time() if now is None else now
        execution.timing.update(
            started_at=now, deadline=now + node.timeout_seconds, interval=node.timeout_seconds
        )


def check_timeout(context, now=None):
    now = time.time() if now is None else now
    arm_timeout(context.node, context.execution, now)
    timing = context.execution.timing
    if "deadline" not in timing or now < timing["deadline"]:
        return
    if timing.get("reported_deadline") != timing["deadline"]:
        record_event(
            context.task, "node.timed_out", {"deadline": timing["deadline"]}, context.execution
        )
        timing["reported_deadline"] = timing["deadline"]
    if not context.node.timeout_human_check:
        raise TimeoutError("Node deadline exceeded; human fallback is disabled.")
    timing.setdefault("resume_status", context.execution.status)
    require_human_check(
        context,
        "timeout",
        {"kind": "timeout", "deadline": timing["deadline"], "node_id": context.node.id},
    )
    timing["deadline"] = now + context.node.timeout_seconds
    record_event(
        context.task, "node.timeout_extended", {"deadline": timing["deadline"]}, context.execution
    )
