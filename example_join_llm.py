"""Portable node settings, LLM prompt delivery, and both join scopes; no service needed."""

import json
from pathlib import Path

from workflow_core import (
    EdgeDefinition,
    NodeDefinition,
    NodeResult,
    Task,
    WorkflowDefinition,
    WorkflowEngine,
)


def build_example(scope):
    examples = Path(__file__).parent / "examples"
    merge = NodeDefinition.load(examples / "nodes" / "merge.json")
    merge.start_condition["scope"] = scope
    workflow = WorkflowDefinition(
        "join_example",
        "Join before LLM",
        [NodeDefinition(name, "step") for name in ("entry", "left", "right")] + [merge],
        [
            EdgeDefinition("entry", "left", when={"exists": "/results/left"}),
            EdgeDefinition("entry", "right", when={"exists": "/results/right"}),
            EdgeDefinition("left", "merge", id="left-to-merge"),
            EdgeDefinition("right", "merge", id="right-to-merge"),
        ],
        ["entry"],
    )
    workflow.assert_valid()
    engine = WorkflowEngine()
    engine.allowances.load(examples / "allowances" / "merge_agent.json")
    engine.register("step", lambda context: NodeResult())
    engine.register_pre_run(
        "inspect_merge",
        lambda ctx, config: {
            "description": config["description"],
            "available": sorted(ctx.inputs["results"]),
        },
    )

    def after(ctx, result, config):
        ctx.execution.local["completion_label"] = config["label"]

    engine.register_post_run("record_completion", after)

    def llm_adapter(messages):
        # A real adapter sends these messages to your model and translates its
        # response into NodeResult or ReActAction. This demo is deterministic.
        prompt = json.loads(messages[1]["content"])
        if prompt.get("request_type") == "active_memory":
            return "Demo merge note: " + prompt["checkpoint"]["name"]
        assert prompt["allowance"]["reference"] == "merge_agent"
        assert prompt["pre_run"][0]["name"] == "inspect_merge"
        assert prompt["post_run"][0]["name"] == "record_completion"
        print("LLM received settings and arrivals:", len(prompt["joined_from"]))
        return NodeResult(values={"summary": prompt["inputs"]["results"]})

    engine.register_llm("agent.merge", llm_adapter)
    return workflow, engine


def main():
    for scope in ("task", "group"):
        workflow, engine = build_example(scope)
        datasets = (
            [{"batch": "case-1", "results": {"left": 1, "right": 2}}]
            if scope == "task"
            else [
                {"batch": "case-1", "results": {"left": 1}},
                {"batch": "case-1", "results": {"right": 2}},
            ]
        )
        for data in datasets:
            engine.task_pool.add(Task.create(workflow, data))
        while engine.task_pool.ready():
            for task, execution in engine.task_pool.ready():
                engine.run_execution(workflow, task, execution)
        completed = [task for task in engine.task_pool.tasks.values() if task.status == "completed"]
        print(scope, "join result:", completed[0].data["summary"])


if __name__ == "__main__":
    main()
