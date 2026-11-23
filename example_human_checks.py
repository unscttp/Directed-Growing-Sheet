"""Interactive framework demo. No external tools or services are called."""

import json
from pathlib import Path

from workflow_core import (
    LocalToolModule,
    NodeResult,
    ReActAction,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
    pending_human_checks,
    resolve_human_check,
)


def reason(context, history):
    if not history:
        return ReActAction("tool:lookup", {"query": "example"})
    return NodeResult(values={"answer": history[-1]["observation"]})


def main():
    examples = Path(__file__).parent / "examples"
    workflow = WorkflowDefinition.load(examples / "human_checks.json")
    tools = LocalToolModule()
    tools.register(
        ToolDefinition.load(examples / "tools" / "reviewed_lookup.json"),
        lambda ctx, tool, arguments: "Example result",
    )
    engine = WorkflowEngine(tools=tools)
    engine.register_react("agent", reason)
    engine.register("done", lambda ctx: NodeResult())
    task = Task.create(workflow)
    while task.active_executions():
        for execution in list(task.executions):
            if execution.status == "ready":
                engine.run_execution(workflow, task, execution)
        for request in pending_human_checks(task):
            print(json.dumps(request, indent=2))
            answer = input("Approve this request? [y/N] ").strip().lower()
            resolve_human_check(task, request["id"], answer == "y", reviewer="local-user")
        if not any(e.status == "ready" for e in task.executions):
            break
    print("Execution statuses:", [(e.node_id, e.status) for e in task.executions])


if __name__ == "__main__":
    main()
