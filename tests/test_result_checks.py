"""react.result_check: the host verifies a node's work before its result counts."""

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


class ResultCheckTests(unittest.TestCase):
    def setUp(self):
        self.done = set()
        tools = LocalToolModule()
        tools.register(
            ToolDefinition("work", "low", input_schema={"type": "object"}),
            lambda context, tool, args: (
                self.done.add(args.get("item")) or {"done": sorted(self.done)}
            ),
        )
        self.engine = WorkflowEngine(tools=tools)
        self.engine.register_result_check(
            "all_done",
            lambda context, result: (
                None
                if {"a", "b"} <= self.done
                else f"Not done yet: {sorted({'a', 'b'} - self.done)}."
            ),
        )
        self.payloads = []

    def run_node(self, replies, react=None):
        replies = list(replies)

        def model(messages):
            payload = json.loads(messages[-1]["content"])
            self.payloads.append(payload)
            return replies.pop(0)

        self.engine.register_llm("agent", model, memory_writer=False)
        node = NodeDefinition.from_dict(
            {
                "id": "n",
                "type": "agent",
                "outputs": {"answer": "/answer"},
                "react": {"max_steps": 4, "result_check": "all_done", **(react or {})},
            }
        )
        workflow = WorkflowDefinition("w", "W", [node], [], ["n"])
        workflow.assert_valid()
        self.task = Task.create(workflow)
        return self.engine.run_execution(workflow, self.task, self.task.executions[0])

    def test_a_rejected_result_goes_back_to_the_model(self):
        result = self.run_node(
            [
                ReActAction("tool:work", {"item": "a"}),
                NodeResult(values={"answer": "finished"}),  # too early
                ReActAction("tool:work", {"item": "b"}),
                NodeResult(values={"answer": "finished"}),
            ]
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.task.data["answer"], "finished")
        rejected = self.payloads[2]["history"][-1]
        self.assertNotIn("action", rejected)
        self.assertEqual(rejected["observation"]["error"]["type"], "ResultRejected")
        self.assertIn("Not done yet: ['b']", rejected["observation"]["error"]["message"])
        events = [e for e in self.task.events if e["type"] == "react.result_rejected"]
        self.assertEqual([e["details"]["check"] for e in events], ["all_done"])

    def test_the_node_fails_when_the_steps_run_out(self):
        result = self.run_node([NodeResult(values={"answer": "done"})] * 4)
        self.assertEqual(result.status, "failed")
        self.assertIn("max_steps", result.error)
        self.assertNotIn("answer", self.task.data)

    def test_the_final_round_is_checked_too(self):
        result = self.run_node(
            [ReActAction("tool:work", {"item": "a"})] * 4
            + [NodeResult(values={"answer": "partial"})],
            react={"on_step_limit": "answer"},
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("result was not accepted: Not done yet: ['b']", result.error)
        self.done.add("b")
        result = self.run_node(
            [ReActAction("tool:work", {"item": "a"})] * 4 + [NodeResult(values={"answer": "all"})],
            react={"on_step_limit": "answer"},
        )
        self.assertEqual(result.status, "completed")

    def test_unknown_or_invalid_checks(self):
        result = self.run_node(
            [NodeResult(values={"answer": "x"})], react={"result_check": "missing"}
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("Unknown result check 'missing'", result.error)
        with self.assertRaises(ValueError):  # the schema refuses an empty name
            NodeDefinition.from_dict({"id": "n", "type": "agent", "react": {"result_check": ""}})
        self.engine.register_result_check("bad", lambda context, result: "")
        result = self.run_node([NodeResult(values={"answer": "x"})], react={"result_check": "bad"})
        self.assertEqual(result.status, "failed")

    def test_generated_graphs_need_permission(self):
        from workflow_graph_tools import _host_references
        from workflow_graphs import DirectedGraph

        graph = DirectedGraph.from_dict(
            {
                "nodes": [{"id": "n", "type": "agent", "react": {"result_check": "all_done"}}],
                "edges": [],
            }
        )
        self.assertIn(("result_check", "all_done"), _host_references(graph))
        self.assertNotIn("all_done", self.engine.generated_graph_names["result_check"])
        self.engine.register_result_check("all_done", lambda c, r: None, generated_graphs=True)
        self.assertIn("all_done", self.engine.generated_graph_names["result_check"])


if __name__ == "__main__":
    unittest.main()
