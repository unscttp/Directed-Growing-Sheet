"""JSON tool descriptions -> filtered LLM prompt -> private tool call. No model service needed."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from workflow_core import (
    BuiltinPrivateToolModule,
    NodeResult,
    ReActAction,
    Task,
    WorkflowDefinition,
    WorkflowEngine,
)


def llm_adapter(messages):
    prompt = json.loads(messages[1]["content"])
    if prompt.get("request_type") == "active_memory":
        return "Demo note for " + prompt["checkpoint"]["name"]
    if not prompt["history"]:
        print("Tools visible to LLM:", [tool["name"] for tool in prompt["available_tools"]])
        return ReActAction("tool:fs.read_text", {"path": "input.txt"})
    return NodeResult(values={"text": prompt["history"][-1]["observation"]["content"]})


def main():
    workflow = WorkflowDefinition.from_dict(
        {
            "id": "private_tools",
            "start": ["read"],
            "edges": [],
            "nodes": [
                {
                    "id": "read",
                    "type": "agent",
                    "allowance": "private_reader",
                    "outputs": {"text": "/text"},
                }
            ],
        }
    )
    with TemporaryDirectory() as directory:
        (Path(directory) / "input.txt").write_text("Private tool result", encoding="utf-8")
        engine = WorkflowEngine(tools=BuiltinPrivateToolModule(directory))
        engine.allowances.load(
            Path(__file__).parent / "examples" / "allowances" / "private_reader.json"
        )
        engine.register_llm("agent", llm_adapter)
        task = Task.create(workflow)
        result = engine.run_execution(workflow, task, task.executions[0])
        print("Execution:", result.status, task.data)


if __name__ == "__main__":
    main()
