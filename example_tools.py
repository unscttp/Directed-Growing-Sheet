"""Run with python example_tools.py; no external service or publishing occurs."""

from pathlib import Path

from workflow_core import (
    AllowanceRegistry,
    LocalToolModule,
    NodeResult,
    ReActAction,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
)


def reason(context, history):
    if not history:
        return ReActAction("tool:documents.lookup", {"query": "example"})
    return NodeResult(values={"answer": history[-1]["observation"]["text"]})


def main():
    examples = Path(__file__).parent / "examples"
    allowances = AllowanceRegistry()
    allowances.load(examples / "allowances" / "low_risk_tools.json")
    tools = LocalToolModule()
    tools.register(
        ToolDefinition.load(examples / "tools" / "lookup.json"),
        lambda context, tool, arguments: {"text": "Example document"},
    )
    tools.register(
        ToolDefinition("documents.publish", "high"),
        lambda context, tool, arguments: {"published": True},
    )
    engine = WorkflowEngine(allowances, tools=tools)
    workflow = WorkflowDefinition.from_dict(
        {
            "id": "tools_example",
            "start": ["lookup"],
            "edges": [],
            "nodes": [
                {
                    "id": "lookup",
                    "type": "agent",
                    "allowance": "low_risk_tools",
                    "outputs": {"answer": "/answer"},
                }
            ],
        }
    )
    engine.register_react("agent", reason)
    task = Task.create(workflow)
    engine.run_execution(workflow, task, task.executions[0])
    print("Allowed:", task.data)

    engine.register(
        "agent",
        lambda context: NodeResult(values={"answer": context.use_tool("documents.publish")}),
    )
    task = Task.create(workflow)
    result = engine.run_execution(workflow, task, task.executions[0])
    print("Blocked:", result.error)


if __name__ == "__main__":
    main()
