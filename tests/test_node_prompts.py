import json
import tempfile
import unittest
from pathlib import Path

from workflow_core import (
    AllowanceSet,
    NodeDefinition,
    NodeResult,
    ReActAction,
    Task,
    WorkflowDefinition,
    WorkflowEngine,
)


class PromptTests(unittest.TestCase):
    def setup_case(self):
        node = NodeDefinition(
            "node",
            "llm",
            allowance="reader",
            pre_run="inspect",
            pre_run_config={"path": "."},
            post_run="finish",
            post_run_config={"format": "text"},
            outputs={"answer": "/answer"},
        )
        workflow = WorkflowDefinition("prompt", "Prompt", [node], [], ["node"])
        engine = WorkflowEngine()
        engine.allowances.register(AllowanceSet("reader", ["read"], max_tool_risk="low"))
        engine.register_pre_run("inspect", lambda ctx, config: {"current": "ready"})
        engine.register_post_run("finish", lambda *args: None)
        return workflow, Task.create(workflow), engine

    def test_standalone_node_and_workflow_json_preserve_settings(self):
        workflow, task, engine = self.setup_case()
        with tempfile.TemporaryDirectory() as directory:
            node_path, workflow_path = (
                Path(directory) / "node.json",
                Path(directory) / "workflow.json",
            )
            workflow.nodes[0].save(node_path)
            workflow.save(workflow_path)
            node = NodeDefinition.load(node_path)
            self.assertEqual(node.to_dict(), workflow.nodes[0].to_dict())
            self.assertEqual(WorkflowDefinition.load(workflow_path).to_dict(), workflow.to_dict())
            self.assertEqual(node.allowance, "reader")
            self.assertEqual(node.pre_run_config, {"path": "."})
            self.assertEqual(node.post_run_config, {"format": "text"})

    def test_llm_actually_receives_configuration_and_context(self):
        workflow, task, engine = self.setup_case()
        captured = []

        def llm(messages):
            self.assertEqual([message["role"] for message in messages], ["system", "user"])
            payload = json.loads(messages[1]["content"])
            captured.append(payload)
            return NodeResult(values={"answer": payload["pre_run_data"]["current"]})

        engine.register_llm("llm", llm, memory_writer=lambda messages: "Observed checkpoint")
        self.assertEqual(
            engine.run_execution(workflow, task, task.executions[0]).status, "completed"
        )
        self.assertEqual(task.data["answer"], "ready")
        self.assertEqual(captured[0]["allowance"]["definition"]["actions"], ["read"])
        self.assertEqual(captured[0]["allowance"]["definition"]["max_tool_risk"], "low")
        self.assertEqual(captured[0]["pre_run"], [{"name": "inspect", "config": {"path": "."}}])
        self.assertEqual(
            captured[0]["post_run"], [{"name": "finish", "config": {"format": "text"}}]
        )

    def test_prompt_refreshes_allowances_each_llm_decision(self):
        workflow, task, engine = self.setup_case()
        captured = []

        def action(ctx, arguments):
            engine.allowances.register(
                AllowanceSet("reader", ["read", "another"], max_tool_risk="medium")
            )
            return "read result"

        engine.register_action("read", action)

        def llm(messages):
            payload = json.loads(messages[1]["content"])
            captured.append(payload)
            return NodeResult() if payload["history"] else ReActAction("read")

        engine.register_llm("llm", llm, memory_writer=lambda messages: "Observed checkpoint")
        engine.run_execution(workflow, task, task.executions[0])
        self.assertEqual(captured[0]["allowance"]["definition"]["max_tool_risk"], "low")
        self.assertEqual(captured[1]["allowance"]["definition"]["max_tool_risk"], "medium")
        self.assertEqual(captured[1]["history"][0]["observation"], "read result")

    def test_ordinary_handlers_can_build_same_prompt(self):
        workflow, task, engine = self.setup_case()

        def handler(ctx):
            payload = ctx.prompt_context
            payload["allowance"]["definition"]["actions"].clear()
            self.assertEqual(ctx.prompt_context["allowance"]["definition"]["actions"], ["read"])
            self.assertEqual(
                json.loads(ctx.prompt_messages()[1]["content"])["node"]["allowance"], "reader"
            )
            return NodeResult()

        engine.register("llm", handler)
        self.assertEqual(
            engine.run_execution(workflow, task, task.executions[0]).status, "completed"
        )

    def test_unset_hooks_and_allowance_explicit_in_prompt(self):
        workflow = WorkflowDefinition(
            "plain", "Plain", [NodeDefinition("node", "llm")], [], ["node"]
        )
        engine = WorkflowEngine()

        def llm(messages):
            payload = json.loads(messages[1]["content"])
            self.assertTrue(payload["allowance"]["unrestricted"])
            self.assertEqual(payload["pre_run"], [])
            self.assertEqual(payload["post_run"], [])
            return NodeResult()

        engine.register_llm("llm", llm, memory_writer=lambda messages: "Observed checkpoint")
        task = Task.create(workflow)
        self.assertEqual(
            engine.run_execution(workflow, task, task.executions[0]).status, "completed"
        )

    def test_invalid_post_run_human_check_is_validation_error(self):
        workflow, task, engine = self.setup_case()
        workflow.nodes[0].post_run_human_check = "yes"
        self.assertIs(workflow.node("node"), workflow.nodes[0])
        with self.assertRaisesRegex(ValueError, "post_run_human_check"):
            workflow.assert_valid()


if __name__ == "__main__":
    unittest.main()
