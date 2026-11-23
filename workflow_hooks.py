"""Optional node lifecycle contracts; applications supply the implementations."""

from __future__ import annotations

from collections.abc import Callable

from workflow_approvals import require_human_check
from workflow_definitions import NodeDefinition
from workflow_nodes import NodeContext, NodeResult
from workflow_types import JsonDict

# Return contextual information for the agent, or None for preparation only.
PreRunHook = Callable[[NodeContext, JsonDict], JsonDict | None]

# Runs after completed outputs are applied and before outgoing edges are checked.
PostRunHook = Callable[[NodeContext, NodeResult, JsonDict], None]


def set_post_run_human_check(node: NodeDefinition, required: bool = True) -> None:
    if type(required) is not bool:
        raise ValueError("Post-run human check must be a Boolean.")
    node.post_run_human_check = required


def require_post_run_human_check(context: NodeContext) -> None:
    """Review completed work and outgoing edge definitions before scheduling."""
    if context.node.post_run_human_check:
        require_human_check(
            context,
            "post_run:edges",
            {
                "kind": "post_run",
                "outputs": context.execution.output_data,
                "data": context.task.data,
                "edges": [edge.to_dict() for edge in context.workflow.outgoing(context.node.id)],
            },
        )
