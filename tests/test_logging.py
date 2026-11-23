import json
import tempfile
import unittest
from pathlib import Path

from workflow_core import (
    AllowanceSet,
    EdgeDefinition,
    LocalToolModule,
    NodeDefinition,
    NodeResult,
    ReActAction,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
    pending_human_checks,
    record_event,
    resolve_human_check,
)


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.workflow = WorkflowDefinition(
            "logs", "Logs", [NodeDefinition("a", "agent")], [], ["a"]
        )
        self.task = Task.create(self.workflow)
        self.tools = LocalToolModule()
        self.tools.register(ToolDefinition("echo", "low"), lambda ctx, tool, args: {"echo": args})
        self.engine = WorkflowEngine(tools=self.tools)

    def context(self):
        return self.engine.build_context(self.workflow, self.task, self.task.executions[0])

    def types(self):
        return [e["type"] for e in self.task.events]

    def test_tool_arguments_results_and_errors_are_correlated(self):
        args = {"value": [1]}
        result = self.context().use_tool("echo", args)
        args["value"].append(2)
        result["echo"]["value"].append(3)
        request, executing, completed = [
            e for e in self.task.events if e["type"].startswith("tool.")
        ][-3:]
        self.assertEqual(request["details"]["arguments"], {"value": [1]})
        self.assertEqual(completed["details"]["result"], {"echo": {"value": [1]}})
        self.assertEqual(executing["details"]["call_id"], request["id"])
        self.assertEqual(completed["details"]["call_id"], request["id"])
        with self.assertRaises(KeyError):
            self.context().use_tool("missing")
        self.assertEqual(
            [e for e in self.task.events if e["type"].startswith("tool.")][-1]["type"],
            "tool.failed",
        )
        self.assertEqual(
            [e for e in self.task.events if e["type"] == "tool.failed"][-1]["details"][
                "error_type"
            ],
            "KeyError",
        )

    def test_denied_invalid_and_backend_failure_calls_logged(self):
        self.engine.allowances.register(AllowanceSet("deny", []))
        self.workflow.nodes[0].allowance = "deny"
        with self.assertRaises(PermissionError):
            self.context().use_tool("echo")
        self.assertNotIn("tool.executing", self.types())
        self.workflow.nodes[0].allowance = None
        with self.assertRaises(ValueError):
            self.context().use_tool("echo", {"bad": float("nan")})
        self.assertEqual(
            [e for e in self.task.events if e["type"] == "tool.requested"][-1]["details"][
                "arguments"
            ],
            {"unserializable_type": "dict"},
        )
        self.tools.register(ToolDefinition("bad", "low"), lambda *args: object())
        with self.assertRaises(TypeError):
            self.context().use_tool("bad")
        self.assertEqual(
            [t for t in self.types() if t.startswith("tool.")][-2:],
            ["tool.executing", "tool.failed"],
        )
        json.dumps(self.task.events, allow_nan=False)

    def test_human_wait_resume_and_decision_are_in_memory(self):
        self.tools.set_human_check("echo")
        self.engine.register_react(
            "agent", lambda ctx, history: NodeResult() if history else ReActAction("tool:echo")
        )
        self.engine.run_execution(self.workflow, self.task, self.task.executions[0])
        self.assertIn("tool.waiting", self.types())
        self.assertNotIn("tool.executing", self.types())
        request = pending_human_checks(self.task)[0]
        resolve_human_check(self.task, request["id"], True, reviewer="tester")
        self.engine.run_execution(self.workflow, self.task, self.task.executions[0])
        self.assertEqual(self.types().count("tool.executing"), 1)
        self.assertIn("approval.resolved", self.types())
        self.assertIn("approval.consumed", self.types())

    def test_llm_receives_previous_node_and_tool_memory(self):
        self.workflow.nodes.append(NodeDefinition("b", "reader"))
        self.workflow.edges.append(EdgeDefinition("a", "b"))
        self.engine.register(
            "agent", lambda ctx: (ctx.use_tool("echo", {"previous": "remember"}), NodeResult())[1]
        )
        self.engine.run_execution(self.workflow, self.task, self.task.executions[0])
        prompts = []

        def llm(messages):
            prompts.append(json.loads(messages[1]["content"]))
            return NodeResult()

        self.engine.register_llm(
            "reader", llm, memory_writer=lambda messages: "Observed checkpoint"
        )
        self.engine.run_execution(self.workflow, self.task, self.task.executions[-1])
        events = prompts[0]["task_memory"]["events"]
        self.assertTrue(
            any(
                e["type"] == "tool.completed"
                and e["details"]["result"] == {"echo": {"previous": "remember"}}
                for e in events
            )
        )
        self.assertTrue(any(e["type"] == "edge.entered" for e in events))
        # Bookkeeping such as llm.started stays in the journal but not in prompts.
        self.assertNotIn("llm.started", {e["type"] for e in events})
        self.assertIn("llm.started", self.types())
        events.clear()
        self.assertTrue(self.task.events)
        self.assertIn("llm.completed", self.types())
        self.assertIn("task.completed", self.types())

    def test_persistence_and_legacy_task(self):
        self.context().use_tool("echo")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            self.task.save(path)
            restored = Task.load(path)
        self.assertEqual(restored.to_dict(), self.task.to_dict())
        raw = self.task.to_dict()
        del raw["events"]
        legacy = Task.from_dict(raw)
        self.assertTrue(legacy.events)
        record_event(restored, "application.note", {"message": "custom activity"})
        self.assertNotEqual(len(restored.events), len(self.task.events))

    def test_hooks_and_node_errors_logged(self):
        self.workflow.nodes[0].pre_run = "prep"
        self.workflow.nodes[0].post_run = "finish"
        self.engine.register_pre_run("prep", lambda *args: {"ready": True})
        self.engine.register_post_run("finish", lambda *args: None)
        self.engine.register("agent", lambda ctx: NodeResult())
        self.engine.run_execution(self.workflow, self.task, self.task.executions[0])
        self.assertEqual(self.types().count("hook.started"), 2)
        self.assertEqual(self.types().count("hook.completed"), 2)
        with self.assertRaises(ValueError):
            self.engine.run_execution(self.workflow, self.task, self.task.executions[0])
        self.assertEqual(self.task.events[-1]["type"], "node.run_error")

    def test_cross_task_join_preserves_source_tool_logs(self):
        self.workflow = WorkflowDefinition(
            "joined",
            "Joined",
            [
                NodeDefinition("left", "branch"),
                NodeDefinition("right", "branch"),
                NodeDefinition(
                    "join",
                    "agent",
                    start_condition={
                        "scope": "group",
                        "required_edges": ["left", "right"],
                        "group_by": "/batch",
                    },
                ),
            ],
            [
                EdgeDefinition("left", "join", id="left"),
                EdgeDefinition("right", "join", id="right"),
            ],
            ["left", "right"],
        )
        sources = []
        self.engine.register(
            "branch", lambda ctx: (ctx.use_tool("echo", {"source": ctx.node.id}), NodeResult())[1]
        )
        for name in ("left", "right"):
            task = Task.create(self.workflow, {"batch": 1})
            task.executions = [e for e in task.executions if e.node_id == name]
            sources.append(task)
            self.engine.run_execution(self.workflow, task, task.executions[0])
        owner, execution = next(
            (t, e) for t, e in self.engine.task_pool.ready() if e.node_id == "join"
        )
        memory = self.engine.build_context(self.workflow, owner, execution).prompt_context[
            "task_memory"
        ]
        completed = [e for e in memory["events"] if e["type"] == "tool.completed"]
        self.assertEqual({e["task_id"] for e in completed}, {t.id for t in sources})
        self.assertEqual(len({e["id"] for e in owner.events}), len(owner.events))


if __name__ == "__main__":
    unittest.main()
