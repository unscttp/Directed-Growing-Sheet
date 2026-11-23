"""Model-independent ReAct loop: decide, dispatch an action, observe, repeat."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from workflow_approvals import HumanCheckRequired
from workflow_logging import record_event
from workflow_nodes import NodeContext, NodeResult
from workflow_types import JsonDict

ACTION_ERROR_MODES = ("observe", "fail")
STEP_LIMIT_MODES = ("fail", "answer")


@dataclass
class ReActAction:
    """One action chosen by the planner.

    ``model_state`` is optional opaque data a model adapter needs to continue
    the conversation after this action, such as a provider's encrypted
    reasoning item or thought signature. The framework stores it with the
    step, hands it back in later prompts' history unshortened, and never
    interprets it or writes it to the task journal.
    """

    name: str
    arguments: JsonDict = field(default_factory=dict)
    model_state: JsonDict | None = None


ReasonHandler = Callable[[NodeContext, list[JsonDict]], "ReActAction | NodeResult"]
ObserveHandler = Callable[[NodeContext, ReActAction, Any], None]


def error_observation(exc: BaseException) -> JsonDict:
    return {"error": {"type": type(exc).__name__, "message": str(exc)}}


def undeclared_outputs(node, result: NodeResult) -> list[str]:
    if result.status != "completed" or not isinstance(result.values, dict):
        return []
    return sorted(set(result.values) - set(node.outputs))


class ReActHandler:
    """Run a reason/action/observation loop for one node execution.

    ``reason(context, history)`` returns a ``ReActAction`` to run, or a
    ``NodeResult`` to finish, wait or fail. Each decision uses one of the
    node's ``react.max_steps`` rounds for this invocation.

    With ``react.on_action_error = "observe"`` (the default), a failed action
    - a denied permission, an unknown tool, a tool exception - is added to the
    history as an error observation so the model can choose differently. A
    final result naming undeclared outputs is handled the same way. With
    ``"fail"``, such errors fail the node. Human-approval pauses and node
    timeouts always stop the loop.

    When ``max_steps`` decisions pass without a result, the node fails
    (``react.on_step_limit = "fail"``, the default). With ``"answer"`` the
    reasoner is asked once more with no tools available and
    ``react_final_round`` set in the prompt, so the model can finish with what
    it has; only if it still asks for an action does the node fail.

    Only actions and observations are persisted (``TaskExecution.local["react"]``),
    never private model reasoning.
    """

    def __init__(self, reason: ReasonHandler, observe: ObserveHandler | None = None) -> None:
        self.reason = reason
        self.observe = observe

    def __call__(self, context: NodeContext) -> NodeResult:
        config = context.node.react
        max_steps = config.get("max_steps", 10)
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("ReAct max_steps must be a positive integer.")
        on_error = config.get("on_action_error", "observe")
        if on_error not in ACTION_ERROR_MODES:
            raise ValueError(f"react.on_action_error must be one of {ACTION_ERROR_MODES}.")
        on_limit = config.get("on_step_limit", "fail")
        if on_limit not in STEP_LIMIT_MODES:
            raise ValueError(f"react.on_step_limit must be one of {STEP_LIMIT_MODES}.")
        history = context.execution.local.setdefault("react", [])
        if not isinstance(history, list):
            raise ValueError("TaskExecution.local['react'] must be a list.")

        for _ in range(max_steps):
            context.check_deadline()
            # Observations of successful calls are persisted before memory is written,
            # so a failing memory writer never causes a tool to run twice.
            context.drain_memory()
            decision = self._next_decision(context, history)
            if isinstance(decision, NodeResult):
                rejected = undeclared_outputs(context.node, decision)
                if not rejected or on_error == "fail":
                    return decision
                error = ValueError(
                    f"Undeclared outputs {rejected}; declared: {sorted(context.node.outputs)}"
                )
                history.append(
                    {"rejected_result": asdict(decision), "observation": error_observation(error)}
                )
                continue
            if not isinstance(decision, ReActAction):
                raise TypeError("ReAct reason must return ReActAction or NodeResult.")
            if decision.model_state is not None and not isinstance(decision.model_state, dict):
                raise TypeError("ReActAction.model_state must be a JSON object or None.")
            self._run_action(context, decision, history, on_error)
        if on_limit == "answer":
            return self._final_round(context, history, max_steps)
        return NodeResult(status="failed", error="ReAct max_steps exceeded.")

    def _final_round(self, context: NodeContext, history, max_steps: int) -> NodeResult:
        """One more decision without tools, so the model can return what it has."""
        record_event(context.task, "react.final_round", {"max_steps": max_steps}, context.execution)
        context.react_final_round = True
        try:
            decision = self.reason(context, copy.deepcopy(history))
            context.record_decision_round()
        finally:
            context.react_final_round = False
        if isinstance(decision, NodeResult) and not undeclared_outputs(context.node, decision):
            return decision
        return NodeResult(
            status="failed",
            error="ReAct max_steps exceeded; the final round returned no valid result.",
        )

    def _next_decision(self, context: NodeContext, history: list[JsonDict]):
        pending = context.execution.local.get("react_pending")
        if pending is not None:
            # Resume exactly the reviewed call instead of asking the model again.
            return ReActAction(**pending)
        decision = self.reason(context, copy.deepcopy(history))
        context.record_decision_round()
        return decision

    def _run_action(self, context, decision: ReActAction, history, on_error: str) -> None:
        succeeded = False
        try:
            observation = context.act(decision.name, decision.arguments)
            succeeded = True
        except (HumanCheckRequired, TimeoutError):
            # Keep the decision so the resumed node runs this exact call.
            context.execution.local["react_pending"] = asdict(decision)
            raise
        except Exception as exc:
            if on_error == "fail":
                context.execution.local.pop("react_pending", None)
                raise
            observation = error_observation(exc)
        context.execution.local.pop("react_pending", None)
        entry = {
            "action": {"name": decision.name, "arguments": decision.arguments},
            "observation": observation,
        }
        if decision.model_state is not None:
            entry["model_state"] = decision.model_state
        json.dumps(entry, allow_nan=False)
        history.append(copy.deepcopy(entry))
        if succeeded and self.observe is not None:
            self.observe(context, decision, copy.deepcopy(observation))
