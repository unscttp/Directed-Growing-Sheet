"""Archived ReAct history: recent and failed steps in full, the rest as an index."""

import json
import unittest

from workflow_core import (
    AllowanceRegistry,
    AllowanceSet,
    LocalToolModule,
    NodeDefinition,
    NodeResult,
    ReActAction,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
)
from workflow_history import archive, parse_steps
from workflow_logging import prompt_size
from workflow_quickstart import _chat_messages


class StepArchiveTests(unittest.TestCase):
    def setUp(self):
        tools = LocalToolModule()

        def work(context, tool, args):
            if args.get("item") == 3:
                raise ValueError("item 3 is broken")
            return {"done": args["item"], "text": "x" * 50}

        tools.register(
            ToolDefinition("work", "low", input_schema={"type": "object"}),
            work,
            summary=lambda args, result: f"did item {args['item']}",
        )
        tools.register(
            ToolDefinition("echo", "low", input_schema={"type": "object"}), lambda *a: "ok"
        )
        allowances = AllowanceRegistry()
        allowances.register(AllowanceSet("worker", ["tool:work", "tool:echo"]))
        self.engine = WorkflowEngine(allowances, tools=tools)
        self.payloads = []
        self.messages = []

    def run_node(self, steps, react=None):
        steps = list(steps)

        def model(messages):
            self.messages.append(messages)
            self.payloads.append(json.loads(messages[-1]["content"]))
            return steps.pop(0)

        self.engine.register_llm("agent", model, memory_writer=False)
        node = NodeDefinition.from_dict(
            {
                "id": "n",
                "type": "agent",
                "allowance": "worker",
                "outputs": {"answer": "/answer"},
                "react": {"max_steps": 30, **(react or {})},
            }
        )
        workflow = WorkflowDefinition("w", "W", [node], [], ["n"])
        self.task = Task.create(workflow)
        return self.engine.run_execution(workflow, self.task, self.task.executions[0])

    def thirteen_steps(self):
        return [ReActAction("tool:work", {"item": i}) for i in range(1, 14)] + [
            NodeResult(values={"answer": "done"})
        ]

    def test_recent_and_failed_steps_in_full_the_rest_indexed(self):
        self.assertEqual(self.run_node(self.thirteen_steps()).status, "completed")
        last = self.payloads[-1]
        self.assertEqual([e["step"] for e in last["history"]], [3, 9, 10, 11, 12, 13])
        self.assertEqual(last["history"][0]["observation"]["error"]["message"], "item 3 is broken")
        lines = last["history_index"]["steps"]
        self.assertEqual(lines[0], "#1 work - did item 1")
        self.assertEqual(lines[2], "#3 work failed (in full below)")
        self.assertEqual(len(lines), 8)
        self.assertEqual(last["history_omitted_older_steps"], 7)
        # history.read is offered once steps are archived, and only then.
        names = lambda payload: [t["name"] for t in payload["available_tools"]]  # noqa: E731
        self.assertNotIn("history.read", names(self.payloads[5]))
        self.assertIn("history.read", names(self.payloads[6]))

    def test_folding_and_per_node_settings(self):
        self.run_node(self.thirteen_steps(), react={"full_steps": 3, "index_fold": 4})
        last = self.payloads[-1]
        self.assertEqual([e["step"] for e in last["history"]], [3, 11, 12, 13])
        self.assertEqual(
            last["history_index"]["steps"],
            [
                "#1-#4: called work ×4; failed: #3 (in full below)",
                "#5-#8: called work ×4",
                "#9 work - did item 9",
                "#10 work - did item 10",
            ],
        )

    def test_index_lines_stay_put_between_requests(self):
        self.run_node(self.thirteen_steps(), react={"index_fold": 0})
        indexes = [p.get("history_index", {}).get("steps", []) for p in self.payloads]
        for before, after in zip(indexes, indexes[1:], strict=False):
            self.assertEqual(after[: len(before)], before)  # only appended to

    def test_history_read(self):
        steps = self.thirteen_steps()
        steps.insert(13, ReActAction("tool:history.read", {"steps": "2,3"}))
        steps.insert(14, ReActAction("tool:history.read", {"steps": "1-11"}))
        steps.insert(15, ReActAction("tool:history.read", {"steps": "40"}))
        self.assertEqual(self.run_node(steps).status, "completed")
        history = self.payloads[-1]["history"]
        read, too_many, missing = (e["observation"] for e in history[-3:])
        self.assertEqual([s["step"] for s in read["steps"]], [2, 3])
        self.assertEqual(read["steps"][0]["observation"]["text"], "x" * 50)  # in full
        self.assertIn("error", read["steps"][1]["observation"])
        self.assertIn("at most 10", too_many["error"]["message"])
        self.assertIn("no step 40", missing["error"]["message"])
        with self.assertRaises(ValueError):
            parse_steps("one", 5)

    def test_the_adapter_shows_the_index(self):
        self.run_node(self.thirteen_steps())
        chat = _chat_messages(self.messages[-1], self.payloads[-1])
        self.assertIn("history_read", chat[1]["content"])
        self.assertIn("#1 work - did item 1", chat[1]["content"])
        call_ids = [m["tool_call_id"] for m in chat if m["role"] == "tool"]
        self.assertEqual(call_ids, ["call_3", "call_9", "call_10", "call_11", "call_12", "call_13"])

    def test_rejected_results_count_as_failures(self):
        history = [
            {"action": {"name": "tool:work", "arguments": {"item": 1}}, "observation": {"done": 1}},
            {
                "rejected_result": {},
                "observation": {"error": {"type": "ResultRejected", "message": "no"}},
            },
            {"action": {"name": "tool:work", "arguments": {"item": 2}}, "observation": {"done": 2}},
        ]
        lines, kept = archive(history, 1, 10)
        self.assertEqual(
            lines, ["#1 work - (item=1) → done=1", "#2 answer not accepted (in full below)"]
        )
        self.assertEqual(kept, [1, 2])

    def test_budget_counts_cjk_characters_like_tokens(self):
        self.assertEqual(prompt_size("abcd"), 4)
        self.assertEqual(prompt_size("文献综述"), 16)


if __name__ == "__main__":
    unittest.main()
