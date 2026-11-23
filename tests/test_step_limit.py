"""react.on_step_limit: a last, tool-less round instead of failing outright."""

import json
import unittest

from workflow_core import (
    LocalToolModule,
    NodeDefinition,
    NodeResult,
    ReActAction,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
)


class StepLimitTests(unittest.TestCase):
    def run_node(self, react, final_reply):
        tools = LocalToolModule()
        tools.register(ToolDefinition("echo", "low"), lambda *args: "ok")
        engine = WorkflowEngine(tools=tools)
        self.payloads = []

        def model(messages):
            payload = json.loads(messages[-1]["content"])
            self.payloads.append(payload)
            if payload.get("react_final_round"):
                return final_reply
            return ReActAction("tool:echo")

        engine.register_llm("agent", model, memory_writer=False)
        node = NodeDefinition.from_dict(
            {"id": "a", "type": "agent", "outputs": {"answer": "/answer"}, "react": react}
        )
        workflow = WorkflowDefinition("w", "W", [node], [], ["a"])
        self.task = Task.create(workflow)
        return engine.run_execution(workflow, self.task, self.task.executions[0])

    def test_default_still_fails(self):
        result = self.run_node({"max_steps": 2}, NodeResult(values={"answer": "x"}))
        self.assertEqual((result.status, result.error), ("failed", "ReAct max_steps exceeded."))
        self.assertEqual(len(self.payloads), 2)

    def test_final_round_answers_without_tools(self):
        result = self.run_node(
            {"max_steps": 2, "on_step_limit": "answer"},
            NodeResult(values={"answer": "best effort"}),
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.task.data["answer"], "best effort")
        final = self.payloads[-1]
        self.assertEqual((final["react_final_round"], final["available_tools"]), (True, []))
        self.assertEqual(len(final["history"]), 2)  # it sees what it did
        self.assertNotIn("react_final_round", self.payloads[0])
        self.assertIn("react.final_round", [e["type"] for e in self.task.events])

    def test_an_action_in_the_final_round_fails(self):
        result = self.run_node(
            {"max_steps": 1, "on_step_limit": "answer"}, ReActAction("tool:echo")
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("final round", result.error)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            NodeDefinition.from_dict(
                {"id": "a", "type": "agent", "react": {"on_step_limit": "retry"}}
            )


if __name__ == "__main__":
    unittest.main()
