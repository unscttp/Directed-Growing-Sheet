"""Run with python example_react.py; no model or external service required."""

from pathlib import Path

from workflow_core import (
    AllowanceRegistry,
    NodeResult,
    ReActAction,
    Task,
    WorkflowDefinition,
    WorkflowEngine,
)


def reason(context, history):
    # Replace this decision hook with your own planner or model adapter.
    if not history:
        return ReActAction("documents.lookup", {"query": context.inputs["query"]})
    return NodeResult(values={"answer": history[-1]["observation"]["text"]})


def main():
    examples = Path(__file__).parent / "examples"
    allowances = AllowanceRegistry()
    allowances.load(examples / "allowances" / "read_only.json")
    workflow = WorkflowDefinition.load(examples / "workflow.json")
    engine = WorkflowEngine(allowances)
    engine.register_action(
        "documents.lookup", lambda context, arguments: {"text": "Example document"}
    )
    engine.register_react("agent.lookup", reason)
    task = Task.create(workflow, {"query": "example"})
    engine.run_execution(workflow, task, task.executions[0])
    print(task.data)
    print(task.executions[0].local["react"])


if __name__ == "__main__":
    main()
