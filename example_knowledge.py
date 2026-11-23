"""Long-term knowledge: a second task uses what the first one learned.

Run ``python example_knowledge.py``. The model is scripted so the example needs
no model service; in an application the three callbacks call your model.
"""

import json
import tempfile
from pathlib import Path

from workflow_core import (
    KnowledgeStore,
    LocalToolModule,
    NodeResult,
    ReActAction,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
)

WORKFLOW = WorkflowDefinition.from_dict(
    {
        "id": "support",
        "name": "Answer a support question",
        "start": ["agent"],
        "edges": [],
        "nodes": [
            {
                "id": "agent",
                "type": "agent",
                "inputs": {"question": "/question"},
                "outputs": {"answer": "/answer"},
                "knowledge": {"read": ["support"], "propose": ["support"]},
            }
        ],
    }
)


def decide(messages):
    """Node decisions: use knowledge when the index has a page, else ask the policy tool."""
    request = json.loads(messages[-1]["content"])
    if request.get("request_type") == "active_memory":
        return "Answered the refund question."
    if request["history"]:
        observation = request["history"][-1]["observation"]
        return NodeResult(values={"answer": observation.get("content") or observation["text"]})
    pages = request["knowledge"]["pages"]
    if pages:
        return ReActAction("tool:knowledge.read", {"page": pages[0]["page"]})
    return ReActAction("tool:policy.lookup", {"topic": "refunds"})


def consolidate(messages):
    """Knowledge writer: propose the fact the task confirmed, citing the tool result."""
    request = json.loads(messages[-1]["content"])
    events = request["task_memory"]["events"]
    lookups = [e["id"] for e in events if e["type"] == "tool.completed"]
    if not lookups or request["knowledge_index"]:
        return {"operations": []}
    return {
        "operations": [
            {
                "op": "create_page",
                "page": "support/refunds",
                "title": "Refunds",
                "summary": "How long refunds take and who approves them",
            },
            {
                "op": "add_entry",
                "page": "support/refunds",
                "text": "Refunds are paid within 5 working days of approval.",
            },
        ],
        "rationale": "The policy lookup confirmed the refund period.",
        "evidence": lookups,
    }


def main():
    with tempfile.TemporaryDirectory() as folder:
        store = KnowledgeStore(Path(folder) / "knowledge")
        tools = LocalToolModule()
        tools.register(
            ToolDefinition("policy.lookup", "low"),
            lambda context, tool, args: {"text": "Policy 4.2: refunds within 5 working days."},
        )
        engine = WorkflowEngine(tools=tools, knowledge=store, knowledge_writer=consolidate)
        engine.register_llm("agent", decide)

        def ask(question):
            task = Task.create(WORKFLOW, {"question": question})
            engine.run_execution(WORKFLOW, task, task.executions[0])
            tools_used = [
                e["details"]["action"] for e in task.events if e["type"] == "tool.requested"
            ]
            print(f"  answer: {task.data['answer']!r}")
            print(f"  tools:  {tools_used}")

        print("Task 1 (nothing known yet)")
        ask("How long does a refund take?")

        (proposal,) = store.proposals()
        print("\nProposal waiting for review")
        source = proposal["source"]
        print(f"  from:      {source['kind']} source, task {source['task_id']}")
        print(f"  outside:   {source['external_actions']}")
        print(f"  rationale: {proposal['rationale']}")
        print(f"  known pages before approval: {len(store.index())}")
        store.resolve_proposal(proposal["id"], True, reviewer="ann", comment="Matches policy 4.2")

        print("\nTask 2 (after a person accepted the proposal)")
        ask("When will my refund arrive?")
        print(f"\nChange log: {[(c['kind'], c['reviewer']) for c in store.changes()]}")


if __name__ == "__main__":
    main()
