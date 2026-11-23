"""Opaque per-action adapter state: persisted, handed back intact, never journaled."""

import json
import unittest

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

SIGNATURE = "sig-" + "x" * 5000  # longer than any prompt shortening limit


def workflow():
    return WorkflowDefinition.from_dict(
        {"id": "w", "start": ["a"], "edges": [], "nodes": [{"id": "a", "type": "agent"}]}
    )


class ModelStateTests(unittest.TestCase):
    def setUp(self):
        self.tools = LocalToolModule()
        self.tools.register(ToolDefinition("lookup", "low"), lambda *a: {"text": "y" * 3000})
        self.tools.register(
            ToolDefinition("guarded", "low", human_check=True), lambda *a: {"ok": True}
        )
        self.engine = WorkflowEngine(tools=self.tools)
        self.prompts = []

    def model(self, *decisions):
        remaining = list(decisions)

        def adapter(messages):
            self.prompts.append(json.loads(messages[-1]["content"]))
            return remaining.pop(0)

        self.engine.register_llm("agent", adapter, memory_writer=False)

    def test_state_is_returned_unshortened_and_kept_out_of_the_journal(self):
        self.model(
            ReActAction("tool:lookup", {}, model_state={"signature": SIGNATURE}),
            NodeResult(),
        )
        task = Task.create(workflow())
        self.engine.run_execution(workflow(), task, task.executions[0])
        (step,) = self.prompts[1]["history"]
        self.assertEqual(step["model_state"], {"signature": SIGNATURE})
        self.assertIn("more characters omitted", step["observation"]["text"])
        self.assertEqual(
            task.executions[0].local["react"][0]["model_state"]["signature"], SIGNATURE
        )
        journal = json.dumps(task.events)
        self.assertNotIn(SIGNATURE, journal)
        completed = [e for e in task.events if e["type"] == "llm.completed"]
        self.assertGreater(completed[0]["details"]["decision"]["model_state_chars"], 5000)

    def test_state_survives_an_approval_pause_and_a_save(self):
        self.model(ReActAction("tool:guarded", {}, model_state={"signature": SIGNATURE}))
        task = Task.create(workflow())
        self.assertEqual(
            self.engine.run_execution(workflow(), task, task.executions[0]).status, "waiting"
        )
        task = Task.from_dict(task.to_dict())
        resolve_human_check(task, pending_human_checks(task)[0]["id"], True, reviewer="me")
        self.model(NodeResult())
        self.engine.run_execution(workflow(), task, task.executions[0])
        self.assertEqual(self.prompts[-1]["history"][0]["model_state"]["signature"], SIGNATURE)

    def test_state_must_be_an_object(self):
        self.model(ReActAction("tool:lookup", {}, model_state="opaque"))
        task = Task.create(workflow())
        result = self.engine.run_execution(workflow(), task, task.executions[0])
        self.assertEqual(result.status, "failed")
        self.assertIn("model_state", result.error)


if __name__ == "__main__":
    unittest.main()
