"""Deterministic mock model demonstrating the new adapter response contracts."""

import json
from pathlib import Path

from workflow_core import NodeResult, Task, WorkflowDefinition, WorkflowEngine


def model(messages):
    payload = json.loads(messages[1]["content"])
    kind = payload.get("request_type", "decision")
    if kind == "active_memory":
        return "Demo operational note for " + payload["checkpoint"]["name"]
    if kind == "branch_selection":
        return {"edges": ["publish"], "reason": "The demo result is ready to publish."}
    return NodeResult()


def main():
    workflow = WorkflowDefinition.load(Path(__file__).parent / "examples" / "runtime_controls.json")
    engine = WorkflowEngine()
    engine.register_llm("agent", model)
    engine.register("worker", lambda ctx: NodeResult())
    task = Task.create(workflow)
    engine.task_pool.add(task)
    while engine.task_pool.ready():
        for owner, execution in engine.task_pool.ready():
            engine.run_execution(workflow, owner, execution)
    print("Task:", task.status)
    print("Objective events:", len(task.events))
    print("Active memories:", [(n["kind"], n["checkpoint"]) for n in task.active_memory["current"]])


if __name__ == "__main__":
    main()
