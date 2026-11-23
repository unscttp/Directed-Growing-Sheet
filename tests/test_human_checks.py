import tempfile
import unittest
from pathlib import Path

from workflow_core import (
    AllowanceSet,
    LocalToolModule,
    NodeResult,
    ReActAction,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
    pending_human_checks,
    require_human_check,
    resolve_human_check,
    set_post_run_human_check,
)


class HumanCheckTests(unittest.TestCase):
    def setup_case(self, tool_check=True, post_check=False):
        workflow = WorkflowDefinition.from_dict(
            {
                "id": "test",
                "start": ["agent"],
                "nodes": [
                    {
                        "id": "agent",
                        "type": "agent",
                        "outputs": {"answer": "/answer"},
                        "post_run_human_check": post_check,
                    },
                    {"id": "next", "type": "next"},
                ],
                "edges": [{"from": "agent", "to": "next"}],
            }
        )
        tools = LocalToolModule()
        calls = []
        tools.register(
            ToolDefinition("read", "low", human_check=tool_check),
            lambda ctx, tool, args: calls.append(args) or {"value": 7},
        )
        engine = WorkflowEngine(tools=tools)
        engine.register_react(
            "agent",
            lambda ctx, history: (
                NodeResult(values={"answer": history[-1]["observation"]})
                if history
                else ReActAction("tool:read", {"query": "test"})
            ),
        )
        return workflow, Task.create(workflow), engine, calls

    def run_node(self, workflow, task, engine):
        return engine.run_execution(workflow, task, task.executions[0])

    def approve(self, task):
        request = pending_human_checks(task)[0]
        resolve_human_check(task, request["id"], True, reviewer="tester")
        return request["id"]

    def test_tool_wait_then_approve_survives_save_load(self):
        workflow, task, engine, calls = self.setup_case()
        self.assertEqual(self.run_node(workflow, task, engine).status, "waiting")
        self.assertEqual(calls, [])
        self.assertEqual(len(task.executions), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            task.save(path)
            task = Task.load(path)
        self.approve(task)
        self.assertEqual(self.run_node(workflow, task, engine).status, "completed")
        self.assertEqual(calls, [{"query": "test"}])
        self.assertEqual(task.executions[0].human_checks[0]["status"], "consumed")
        self.assertEqual(task.executions[1].node_id, "next")

    def test_rejection_stops_execution(self):
        for tool_check, post_check in ((True, False), (False, True)):
            workflow, task, engine, calls = self.setup_case(tool_check, post_check)
            self.run_node(workflow, task, engine)
            request = pending_human_checks(task)[0]
            resolve_human_check(task, request["id"], False, reviewer="tester", comment="No")
            self.assertEqual(task.executions[0].status, "failed")
            with self.assertRaises(ValueError):
                self.run_node(workflow, task, engine)
            self.assertEqual(len(task.executions), 1)
            self.assertEqual(len(calls), 0 if tool_check else 1)

    def test_post_resume_does_not_repeat_agent_or_hook(self):
        workflow, task, engine, calls = self.setup_case(False, True)
        hooks = []
        workflow.nodes[0].post_run = "finish"
        engine.register_post_run("finish", lambda *args: hooks.append("post"))
        self.assertEqual(self.run_node(workflow, task, engine).status, "waiting")
        task = Task.from_dict(task.to_dict())
        self.approve(task)
        self.assertEqual(self.run_node(workflow, task, engine).status, "completed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(hooks, ["post"])
        self.assertEqual(task.executions[0].continuation, {})

    def test_tool_and_post_are_independent_reviews(self):
        workflow, task, engine, calls = self.setup_case(True, True)
        self.run_node(workflow, task, engine)
        self.approve(task)
        self.assertEqual(self.run_node(workflow, task, engine).status, "waiting")
        self.assertEqual(pending_human_checks(task)[0]["details"]["kind"], "post_run")
        self.approve(task)
        self.assertEqual(self.run_node(workflow, task, engine).status, "completed")
        self.assertEqual(len(calls), 1)

    def test_changed_arguments_need_new_review(self):
        workflow, task, engine, calls = self.setup_case()
        arguments = {"value": 1}
        engine.register(
            "agent", lambda ctx: NodeResult(values={"answer": ctx.use_tool("read", arguments)})
        )
        self.run_node(workflow, task, engine)
        self.approve(task)
        arguments["value"] = True  # Boolean and integer must not share approval.
        self.assertEqual(self.run_node(workflow, task, engine).status, "waiting")
        self.assertEqual(calls, [])
        self.assertEqual(task.executions[0].human_checks[0]["status"], "superseded")

    def test_changed_post_data_or_edges_need_new_review(self):
        for change in ("data", "edges"):
            workflow, task, engine, calls = self.setup_case(False, True)
            self.run_node(workflow, task, engine)
            self.approve(task)
            if change == "data":
                task.data["new"] = True
            else:
                workflow.edges[0].label = "changed route"
            self.assertEqual(self.run_node(workflow, task, engine).status, "waiting")
            self.assertEqual(len(task.executions), 1)
            self.assertEqual(len(calls), 1)

    def test_allowances_rechecked_after_approval(self):
        workflow, task, engine, calls = self.setup_case()
        workflow.nodes[0].allowance = "policy"
        workflow.nodes[0].react = {"on_action_error": "fail"}
        engine.allowances.register(AllowanceSet("policy", ["tool:read"]))
        self.run_node(workflow, task, engine)
        self.approve(task)
        engine.allowances.register(AllowanceSet("policy", []))
        self.assertEqual(self.run_node(workflow, task, engine).status, "failed")
        self.assertEqual(calls, [])

    def test_each_repeated_tool_call_needs_own_approval(self):
        workflow, task, engine, calls = self.setup_case()
        engine.register_react(
            "agent",
            lambda ctx, history: NodeResult() if len(history) == 2 else ReActAction("tool:read"),
        )
        for _ in range(2):
            self.assertEqual(self.run_node(workflow, task, engine).status, "waiting")
            self.approve(task)
        self.assertEqual(self.run_node(workflow, task, engine).status, "completed")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(task.executions[0].human_checks), 2)

    def test_pending_react_action_does_not_replan(self):
        workflow, task, engine, calls = self.setup_case()
        reasons = []

        def reason(ctx, history):
            reasons.append(True)
            return NodeResult() if history else ReActAction("tool:read")

        engine.register_react("agent", reason)
        self.run_node(workflow, task, engine)
        self.approve(task)
        self.run_node(workflow, task, engine)
        self.assertEqual(len(reasons), 2)
        self.assertNotIn("react_pending", task.executions[0].local)

    def test_custom_post_checkpoint(self):
        workflow, task, engine, calls = self.setup_case(False)
        workflow.nodes[0].post_run = "review"
        effects = []

        def review(ctx, result, config):
            require_human_check(ctx, "review", {"result": result.values})
            effects.append("done")

        engine.register_post_run("review", review)
        self.assertEqual(self.run_node(workflow, task, engine).status, "waiting")
        self.assertEqual(effects, [])
        self.approve(task)
        self.run_node(workflow, task, engine)
        self.assertEqual(effects, ["done"])
        self.assertEqual(len(calls), 1)

    def test_edge_check_still_applies_after_human_approval(self):
        workflow, task, engine, calls = self.setup_case(False, True)
        workflow.edges[0].entry_check = "deny"
        engine.register_entry_check("deny", lambda *args: False)
        self.run_node(workflow, task, engine)
        self.approve(task)
        self.run_node(workflow, task, engine)
        self.assertEqual(len(task.executions), 2)
        self.assertEqual(task.executions[-1].status, "skipped")

    def test_invalid_and_duplicate_decisions(self):
        workflow, task, engine, calls = self.setup_case()
        self.run_node(workflow, task, engine)
        request = pending_human_checks(task)[0]
        with self.assertRaises(TypeError):
            resolve_human_check(task, request["id"], "yes", reviewer="tester")
        with self.assertRaises(ValueError):
            resolve_human_check(task, request["id"], True, reviewer="")
        with self.assertRaises(KeyError):
            resolve_human_check(task, "missing", True, reviewer="tester")
        self.approve(task)
        with self.assertRaises(ValueError):
            resolve_human_check(task, request["id"], True, reviewer="tester")

    def test_settings_json_and_setters(self):
        workflow, task, engine, calls = self.setup_case(False)
        engine.tools.set_human_check("read")
        set_post_run_human_check(workflow.nodes[0])
        self.assertTrue(
            ToolDefinition.from_dict(engine.tools.get_tool("read").to_dict()).human_check
        )
        self.assertTrue(
            WorkflowDefinition.from_dict(workflow.to_dict()).nodes[0].post_run_human_check
        )
        with self.assertRaises(ValueError):
            ToolDefinition("bad", "low", human_check="yes")
        with self.assertRaises(ValueError):
            set_post_run_human_check(workflow.nodes[0], "yes")

    def test_custom_backend_is_also_gated(self):
        workflow, task, engine, calls = self.setup_case()

        class Backend:
            def get_tool(self, name):
                return ToolDefinition(name, "low", human_check=True)

            def execute(self, context, tool, arguments):
                calls.append("custom")
                return "done"

        engine.attach_tools(Backend())
        self.assertEqual(self.run_node(workflow, task, engine).status, "waiting")
        self.assertEqual(calls, [])
        self.approve(task)
        self.assertEqual(self.run_node(workflow, task, engine).status, "completed")
        self.assertEqual(calls, ["custom"])

    def test_post_run_tool_pause_does_not_repeat_agent(self):
        workflow, task, engine, calls = self.setup_case()
        agents = []
        engine.register("agent", lambda ctx: agents.append("agent") or NodeResult())
        workflow.nodes[0].post_run = "use_tool"

        def post_run(ctx, result, config):
            ctx.use_tool("read")

        engine.register_post_run("use_tool", post_run)
        self.assertEqual(self.run_node(workflow, task, engine).status, "waiting")
        self.assertEqual(calls, [])
        task = Task.from_dict(task.to_dict())
        self.approve(task)
        self.assertEqual(self.run_node(workflow, task, engine).status, "completed")
        self.assertEqual(agents, ["agent"])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
