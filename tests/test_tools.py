import tempfile
import unittest
from pathlib import Path

from workflow_core import (
    RISK_LEVELS,
    AllowanceRegistry,
    AllowanceSet,
    LocalToolModule,
    NodeResult,
    ReActAction,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
)


class ToolTests(unittest.TestCase):
    def setup_engine(self, policy=None, risk="low", allowance_check=None):
        workflow = WorkflowDefinition.from_dict(
            {
                "id": "tools",
                "start": ["node"],
                "edges": [],
                "nodes": [
                    {
                        "id": "node",
                        "type": "agent",
                        "outputs": {"answer": "/answer"},
                        "allowance": policy.id if policy else None,
                    }
                ],
            }
        )
        registry = AllowanceRegistry()
        if policy:
            registry.register(policy)
        calls = []
        tools = LocalToolModule()

        def execute(context, tool, arguments):
            calls.append((tool.name, arguments))
            return {"value": 7}

        tools.register(ToolDefinition("read", risk), execute)
        engine = WorkflowEngine(registry, allowance_check, tools)
        engine.register("agent", lambda ctx: NodeResult(values={"answer": ctx.use_tool("read")}))
        return workflow, engine, tools, calls

    def run_node(self, workflow, engine):
        task = Task.create(workflow)
        result = engine.run_execution(workflow, task, task.executions[0])
        return task, result

    def test_all_risk_boundaries(self):
        for ceiling_index, ceiling in enumerate(RISK_LEVELS):
            for risk_index, risk in enumerate(RISK_LEVELS):
                with self.subTest(ceiling=ceiling, risk=risk):
                    workflow, engine, tools, calls = self.setup_engine(
                        AllowanceSet("policy", ["tool:read"], max_tool_risk=ceiling), risk
                    )
                    task, result = self.run_node(workflow, engine)
                    allowed = risk_index <= ceiling_index
                    self.assertEqual(result.status, "completed" if allowed else "failed")
                    self.assertEqual(len(calls), int(allowed))
                    if not allowed:
                        self.assertIn("PermissionError", result.error)

    def test_name_and_custom_checks_still_required(self):
        for actions, callback in (
            ([], None),
            (["read"], None),
            (["tool:read"], lambda *args: False),
        ):
            with self.subTest(actions=actions):
                workflow, engine, tools, calls = self.setup_engine(
                    AllowanceSet("policy", actions, max_tool_risk="critical"),
                    allowance_check=callback,
                )
                self.assertEqual(self.run_node(workflow, engine)[1].status, "failed")
                self.assertEqual(calls, [])

    def test_optional_policy_and_optional_ceiling(self):
        for policy in (None, AllowanceSet("legacy", ["tool:read"])):
            workflow, engine, tools, calls = self.setup_engine(policy, risk="critical")
            self.assertEqual(self.run_node(workflow, engine)[1].status, "completed")
            self.assertEqual(len(calls), 1)

    def test_react_tool_result_is_persisted(self):
        workflow, engine, tools, calls = self.setup_engine(
            AllowanceSet("policy", ["tool:read"], max_tool_risk="low")
        )
        engine.register_react(
            "agent",
            lambda ctx, history: (
                NodeResult(values={"answer": history[-1]["observation"]})
                if history
                else ReActAction("tool:read", {"query": "test"})
            ),
        )
        task, result = self.run_node(workflow, engine)
        self.assertEqual(result.status, "completed")
        restored = Task.from_dict(task.to_dict())
        self.assertEqual(restored.executions[0].local["react"][0]["observation"], {"value": 7})
        self.assertEqual(calls, [("read", {"query": "test"})])

    def test_custom_backend_and_replacement(self):
        class Backend:
            def __init__(self, risk):
                self.risk, self.calls = risk, []

            def get_tool(self, name):
                return ToolDefinition(name, self.risk)

            def execute(self, context, tool, arguments):
                self.calls.append(tool.name)
                return "custom"

        workflow, engine, tools, calls = self.setup_engine(
            AllowanceSet("policy", ["tool:read"], max_tool_risk="low")
        )
        backend = Backend("low")
        engine.attach_tools(backend)
        self.assertEqual(self.run_node(workflow, engine)[0].data["answer"], "custom")
        self.assertEqual(backend.calls, ["read"])
        replacement = Backend("high")
        engine.attach_tools(replacement)
        self.assertEqual(self.run_node(workflow, engine)[1].status, "failed")
        self.assertEqual(replacement.calls, [])
        self.assertEqual(calls, [])

    def test_json_round_trip(self):
        policy = AllowanceSet("policy", ["tool:read"], max_tool_risk="medium")
        tool = ToolDefinition("read", "medium", config={"provider": "custom"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy.save(root / "policy.json")
            tool.save(root / "tool.json")
            self.assertEqual(AllowanceSet.load(root / "policy.json"), policy)
            self.assertEqual(ToolDefinition.load(root / "tool.json"), tool)

    def test_mutable_settings_revalidated_and_updates_applied(self):
        workflow, engine, tools, calls = self.setup_engine(
            AllowanceSet("policy", ["tool:read"], max_tool_risk="low"), "high"
        )
        self.assertEqual(self.run_node(workflow, engine)[1].status, "failed")
        policy = engine.allowances.resolve("policy")
        policy.max_tool_risk = "high"
        engine.allowances.register(policy)
        self.assertEqual(self.run_node(workflow, engine)[1].status, "completed")
        tools.register(ToolDefinition("read", "critical"), lambda *args: calls.append("unexpected"))
        self.assertEqual(self.run_node(workflow, engine)[1].status, "failed")
        self.assertEqual(len(calls), 1)
        policy.max_tool_risk = "invalid"
        with self.assertRaises(ValueError):
            engine.allowances.register(policy)

    def test_invalid_risk_values(self):
        for risk in (None, True, 1, "", "unknown", [], {}):
            with self.subTest(risk=risk), self.assertRaises(ValueError):
                ToolDefinition("read", risk)
            if risk is not None:
                with self.subTest(ceiling=risk), self.assertRaises(ValueError):
                    AllowanceSet("policy", max_tool_risk=risk)
        with self.assertRaises(ValueError):
            ToolDefinition.from_dict({"name": "read"})

    def test_detachment_unknown_tools_and_reserved_prefix(self):
        workflow, engine, tools, calls = self.setup_engine()
        engine.attach_tools(None)
        self.assertIn("No tool module", self.run_node(workflow, engine)[1].error)
        engine.attach_tools(LocalToolModule())
        self.assertIn("Unknown tool", self.run_node(workflow, engine)[1].error)
        with self.assertRaises(ValueError):
            engine.register_action("tool:read", lambda *args: True)
        with self.assertRaises(TypeError):
            engine.attach_tools(object())
        self.assertEqual(calls, [])

    def test_tool_errors_and_non_json_results(self):
        workflow, engine, tools, calls = self.setup_engine()

        def broken(*args):
            raise RuntimeError("tool failed")

        for handler, error in ((broken, "RuntimeError"), (lambda *args: object(), "TypeError")):
            tools.register(ToolDefinition("read", "low"), handler)
            self.assertIn(error, self.run_node(workflow, engine)[1].error)

    def test_risk_cannot_be_overridden_by_arguments(self):
        workflow, engine, tools, calls = self.setup_engine(
            AllowanceSet("policy", ["tool:read"], max_tool_risk="low"), "high"
        )
        engine.register(
            "agent",
            lambda ctx: NodeResult(values={"answer": ctx.use_tool("read", {"risk_level": "low"})}),
        )
        self.assertEqual(self.run_node(workflow, engine)[1].status, "failed")
        self.assertEqual(calls, [])

    def test_bad_backend_metadata_cannot_execute(self):
        class Backend:
            def get_tool(self, name):
                return ToolDefinition("different", "low")

            def execute(self, *args):
                raise AssertionError("Must not execute")

        workflow, engine, tools, calls = self.setup_engine()
        engine.attach_tools(Backend())
        self.assertIn("different tool name", self.run_node(workflow, engine)[1].error)

    def test_ordinary_actions_unaffected_by_tool_ceiling(self):
        workflow, engine, tools, calls = self.setup_engine(
            AllowanceSet("policy", ["ordinary"], max_tool_risk="low")
        )
        engine.register_action("ordinary", lambda ctx, args: 1)
        engine.register("agent", lambda ctx: NodeResult(values={"answer": ctx.act("ordinary")}))
        self.assertEqual(self.run_node(workflow, engine)[0].data["answer"], 1)


if __name__ == "__main__":
    unittest.main()
