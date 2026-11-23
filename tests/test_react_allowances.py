import tempfile
import unittest
from pathlib import Path

from workflow_core import (
    AllowanceRegistry,
    AllowanceSet,
    NodeResult,
    ReActAction,
    Task,
    WorkflowDefinition,
    WorkflowEngine,
)


class FrameworkTests(unittest.TestCase):
    def setup_node(self, allowance=None, max_steps=4, on_action_error=None):
        workflow = WorkflowDefinition.from_dict(
            {
                "id": "test",
                "start": ["node"],
                "edges": [],
                "nodes": [
                    {
                        "id": "node",
                        "type": "agent",
                        "outputs": {"answer": "/answer"},
                        "allowance": allowance,
                        "react": {"max_steps": max_steps}
                        | ({"on_action_error": on_action_error} if on_action_error else {}),
                    }
                ],
            }
        )
        return workflow, Task.create(workflow)

    def run_node(self, engine, workflow, task):
        return engine.run_execution(workflow, task, task.executions[0])

    def test_existing_handler(self):
        workflow = WorkflowDefinition.from_dict(
            {
                "id": "old",
                "start": ["node"],
                "edges": [],
                "nodes": [{"id": "node", "type": "old"}],
            }
        )
        task = Task.create(workflow)
        engine = WorkflowEngine()
        engine.register("old", lambda ctx: NodeResult())
        self.assertEqual(self.run_node(engine, workflow, task).status, "completed")
        self.assertEqual(task.status, "completed")
        self.assertEqual(task.executions[0].local, {})

    def test_react_cycle_and_persistence(self):
        workflow, task = self.setup_node("reader")
        registry = AllowanceRegistry()
        registry.register(AllowanceSet("reader", ["read"]))
        engine = WorkflowEngine(registry)
        observed = []
        engine.register_action("read", lambda ctx, args: {"value": args["value"]})
        engine.register_react(
            "agent",
            lambda ctx, history: (
                NodeResult(values={"answer": history[-1]["observation"]["value"]})
                if history
                else ReActAction("read", {"value": 7})
            ),
            lambda ctx, action, observation: observed.append(observation),
        )
        self.assertEqual(self.run_node(engine, workflow, task).status, "completed")
        self.assertEqual(task.data, {"answer": 7})
        self.assertEqual(observed, [{"value": 7}])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow.save(root / "workflow.json")
            task.save(root / "task.json")
            registry.resolve("reader").save(root / "allowance.json")
            self.assertEqual(
                WorkflowDefinition.load(root / "workflow.json").to_dict(), workflow.to_dict()
            )
            self.assertEqual(Task.load(root / "task.json").to_dict(), task.to_dict())
            self.assertEqual(AllowanceRegistry().load(root / "allowance.json").actions, ["read"])

    def test_denied_action_never_runs(self):
        for allowed in ([], ["read"]):
            for mode in ("observe", "fail"):
                with self.subTest(allowed=allowed, mode=mode):
                    workflow, task = self.setup_node("reader", on_action_error=mode)
                    registry = AllowanceRegistry()
                    registry.register(AllowanceSet("reader", allowed))
                    engine = WorkflowEngine(registry)
                    called = []
                    engine.register_action("write", lambda ctx, args: called.append(True))
                    engine.register_react("agent", lambda ctx, history: ReActAction("write"))
                    result = self.run_node(engine, workflow, task)
                    self.assertEqual(result.status, "failed")
                    self.assertEqual(called, [])
                    if mode == "fail":
                        self.assertIn("PermissionError", result.error)
                    else:
                        # Each denial is shown to the planner, which never adapts here.
                        self.assertEqual(result.error, "ReAct max_steps exceeded.")
                        history = task.executions[0].local["react"]
                        errors = {entry["observation"]["error"]["type"] for entry in history}
                        self.assertEqual(errors, {"PermissionError"})

    def test_planner_recovers_from_failed_action(self):
        workflow, task = self.setup_node()
        engine = WorkflowEngine()
        engine.register_action("read", lambda ctx, args: {"text": args["path"]})

        def reason(ctx, history):
            if not history:
                return ReActAction("missing.tool")
            if len(history) == 1:
                self.assertEqual(history[0]["observation"]["error"]["type"], "KeyError")
                return ReActAction("read", {"path": "ok.txt"})
            return NodeResult(values={"answer": history[-1]["observation"]["text"]})

        engine.register_react("agent", reason)
        self.assertEqual(self.run_node(engine, workflow, task).status, "completed")
        self.assertEqual(task.data["answer"], "ok.txt")

    def test_undeclared_final_outputs_are_returned_to_planner(self):
        workflow, task = self.setup_node()
        engine = WorkflowEngine()

        def reason(ctx, history):
            if not history:
                return NodeResult(values={"invented": 1})
            self.assertIn("Undeclared outputs", history[0]["observation"]["error"]["message"])
            return NodeResult(values={"answer": 2})

        engine.register_react("agent", reason)
        self.assertEqual(self.run_node(engine, workflow, task).status, "completed")
        self.assertEqual(task.data, {"answer": 2})

    def test_missing_reference_prevents_handler(self):
        workflow, task = self.setup_node("missing")
        called = []
        engine = WorkflowEngine()
        engine.register("agent", lambda ctx: called.append(True))
        result = self.run_node(engine, workflow, task)
        self.assertIn("Unknown allowance set", result.error)
        self.assertEqual(called, [])

    def test_optional_allowance_and_ordinary_action_dispatch(self):
        workflow, task = self.setup_node()
        engine = WorkflowEngine()
        engine.register_action("read", lambda ctx, args: 42)
        engine.register("agent", lambda ctx: NodeResult(values={"answer": ctx.act("read")}))
        self.run_node(engine, workflow, task)
        self.assertEqual(task.data["answer"], 42)

    def test_custom_rule_blocks_before_action(self):
        workflow, task = self.setup_node("reader")
        registry = AllowanceRegistry()
        registry.register(AllowanceSet("reader", ["read"], config={"limit": 3}))
        engine = WorkflowEngine(
            registry,
            allowance_check=lambda ctx, policy, action, args: (
                args["count"] <= policy.config["limit"]
            ),
        )
        called = []
        engine.register_action("read", lambda ctx, args: called.append(True))
        engine.register_react("agent", lambda ctx, history: ReActAction("read", {"count": 4}))
        self.assertEqual(self.run_node(engine, workflow, task).status, "failed")
        self.assertEqual(called, [])

    def test_replacement_applies_to_next_action(self):
        workflow, task = self.setup_node("reader")
        registry = AllowanceRegistry()
        registry.register(AllowanceSet("reader", ["read"]))
        engine = WorkflowEngine(registry)
        called = []
        engine.register_action("read", lambda ctx, args: called.append(True))
        engine.register_react(
            "agent",
            lambda ctx, history: ReActAction("read"),
            lambda *args: registry.register(AllowanceSet("reader")),
        )
        self.assertEqual(self.run_node(engine, workflow, task).status, "failed")
        self.assertEqual(called, [True])

    def test_step_limit(self):
        workflow, task = self.setup_node(max_steps=2)
        engine = WorkflowEngine()
        engine.register_action("read", lambda ctx, args: 1)
        engine.register_react("agent", lambda ctx, history: ReActAction("read"))
        result = self.run_node(engine, workflow, task)
        self.assertEqual(result.error, "ReAct max_steps exceeded.")
        self.assertEqual(len(task.executions[0].local["react"]), 2)

    def test_wait_save_resume(self):
        workflow, task = self.setup_node()
        engine = WorkflowEngine()
        engine.register_action("read", lambda ctx, args: "done")

        def reason(ctx, history):
            if not history:
                return ReActAction("read")
            if not ctx.execution.local.get("approved"):
                return NodeResult(status="waiting")
            return NodeResult(values={"answer": history[-1]["observation"]})

        engine.register_react("agent", reason)
        self.assertEqual(self.run_node(engine, workflow, task).status, "waiting")
        task = Task.from_dict(task.to_dict())
        task.executions[0].local["approved"] = True
        task.executions[0].status = "ready"
        self.assertEqual(self.run_node(engine, workflow, task).status, "completed")
        self.assertEqual(len(task.executions[0].local["react"]), 1)

    def test_bad_configuration(self):
        for limit in (0, -1, True, "2"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.setup_node(max_steps=limit)
        for actions in ("read", [1], [""]):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                AllowanceSet.from_dict({"id": "bad", "actions": actions})

    def test_unknown_action_and_action_error(self):
        for registered in (False, True):
            with self.subTest(registered=registered):
                workflow, task = self.setup_node(on_action_error="fail")
                engine = WorkflowEngine()

                def broken(ctx, args):
                    raise RuntimeError("action failed")

                if registered:
                    engine.register_action("broken", broken)
                engine.register_react("agent", lambda ctx, history: ReActAction("broken"))
                result = self.run_node(engine, workflow, task)
                self.assertEqual(result.status, "failed")
                self.assertIn("RuntimeError" if registered else "KeyError", result.error)


if __name__ == "__main__":
    unittest.main()
