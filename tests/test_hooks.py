import tempfile
import unittest
from pathlib import Path

from workflow_core import (
    AllowanceSet,
    LocalToolModule,
    NodeResult,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
)


class HookTests(unittest.TestCase):
    def setup_node(self, **fields):
        workflow = WorkflowDefinition.from_dict(
            {
                "id": "hooks",
                "start": ["agent"],
                "nodes": [
                    {"id": "agent", "type": "agent", "outputs": {"answer": "/answer"}, **fields},
                    {"id": "next", "type": "agent"},
                ],
                "edges": [{"from": "agent", "to": "next"}],
            }
        )
        engine = WorkflowEngine()
        engine.register("agent", lambda ctx: NodeResult(values={"answer": 7}))
        return workflow, Task.create(workflow), engine

    def run_node(self, workflow, task, engine):
        return engine.run_execution(workflow, task, task.executions[0])

    def test_optional_hooks_leave_existing_behavior(self):
        for fields in ({}, {"pre_run": None, "post_run": None}):
            workflow, task, engine = self.setup_node(**fields)
            self.assertEqual(self.run_node(workflow, task, engine).status, "completed")
            self.assertEqual(task.executions[0].local, {})
            self.assertEqual(task.executions[1].node_id, "next")

    def test_lifecycle_order_and_agent_information(self):
        workflow, task, engine = self.setup_node(pre_run="prepare", post_run="finish")
        events = []

        def prepare(ctx, config):
            events.append("pre")
            ctx.inputs["extra"] = 3
            return {"situation": "ready"}

        def agent(ctx):
            events.append("agent")
            self.assertEqual(ctx.pre_run_data, {"situation": "ready"})
            self.assertEqual(ctx.inputs["extra"], 3)
            return NodeResult(values={"answer": 7})

        def finish(ctx, result, config):
            events.append("post")
            self.assertEqual(ctx.task.data["answer"], 7)
            self.assertEqual(ctx.execution.output_data, {"answer": 7})
            self.assertEqual(result.values["answer"], 7)

        def check(ctx, config):
            events.append("edge")
            return True

        engine.register_pre_run("prepare", prepare)
        engine.register("agent", agent)
        engine.register_post_run("finish", finish)
        workflow.edges[0].entry_check = "check"
        engine.register_entry_check("check", check)
        self.run_node(workflow, task, engine)
        self.assertEqual(events, ["pre", "agent", "post", "edge"])
        self.assertEqual(task.executions[0].input_data, {"extra": 3})
        self.assertEqual(
            Task.from_dict(task.to_dict()).executions[0].local["pre_run"], {"situation": "ready"}
        )

    def test_react_receives_pre_run_data(self):
        workflow, task, engine = self.setup_node(pre_run="prepare")
        engine.register_pre_run("prepare", lambda ctx, config: {"value": 9})
        engine.register_react(
            "agent", lambda ctx, history: NodeResult(values={"answer": ctx.pre_run_data["value"]})
        )
        self.run_node(workflow, task, engine)
        self.assertEqual(task.data["answer"], 9)

    def test_post_run_file_available_to_edge_check(self):
        workflow, task, engine = self.setup_node(post_run="write")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.txt"

            def write(ctx, result, config):
                path.write_text(str(result.values["answer"]), encoding="utf-8")

            engine.register_post_run("write", write)
            workflow.edges[0].entry_check = "exists"
            engine.register_entry_check("exists", lambda ctx, config: path.is_file())
            self.run_node(workflow, task, engine)
            self.assertEqual(path.read_text(encoding="utf-8"), "7")
            self.assertEqual(task.executions[1].node_id, "next")

    def test_waiting_and_failure_skip_post_run(self):
        for status in ("waiting", "failed", "raises"):
            with self.subTest(status=status):
                workflow, task, engine = self.setup_node(post_run="finish")
                calls = []
                engine.register_post_run("finish", lambda *args: calls.append(True))

                def agent(ctx):
                    if status == "raises":
                        raise RuntimeError("failed")
                    return NodeResult(status=status)

                engine.register("agent", agent)
                self.run_node(workflow, task, engine)
                self.assertEqual(calls, [])
                self.assertEqual(len(task.executions), 1)

    def test_pre_run_failures_prevent_agent(self):
        def broken(ctx, config):
            raise RuntimeError("preparation failed")

        for hook in (broken, lambda c, p: "bad", lambda c, p: {"bad": object()}):
            workflow, task, engine = self.setup_node(pre_run="prepare")
            calls = []
            engine.register("agent", lambda ctx: calls.append(True))
            engine.register_pre_run("prepare", hook)
            self.assertEqual(self.run_node(workflow, task, engine).status, "failed")
            self.assertEqual(calls, [])
            self.assertEqual(len(task.executions), 1)

    def test_post_run_failure_keeps_outputs_but_blocks_edges(self):
        def broken(ctx, result, config):
            raise RuntimeError("automatic work failed")

        for hook in (broken, lambda *args: {"bad": "return"}):
            workflow, task, engine = self.setup_node(post_run="finish")
            engine.register_post_run("finish", hook)
            result = self.run_node(workflow, task, engine)
            self.assertEqual(result.status, "failed")
            self.assertIn("Post-run hook", result.error)
            self.assertEqual(task.data["answer"], 7)
            self.assertEqual(task.executions[0].status, "failed")
            self.assertEqual(len(task.executions), 1)

    def test_both_references_resolved_before_preparation(self):
        for missing in ("pre_run", "post_run"):
            workflow, task, engine = self.setup_node(**{missing: "missing"})
            calls = []
            if missing == "post_run":
                workflow.nodes[0].pre_run = "prepare"
                engine.register_pre_run("prepare", lambda *args: calls.append("pre"))
            engine.register("agent", lambda ctx: calls.append("agent"))
            self.assertEqual(self.run_node(workflow, task, engine).status, "failed")
            self.assertEqual(calls, [])

    def test_json_round_trip_and_copy_isolation(self):
        workflow, task, engine = self.setup_node(
            pre_run="prepare",
            pre_run_config={"items": []},
            post_run="finish",
            post_run_config={"items": []},
        )
        information = {"items": []}

        def prepare(ctx, config):
            config["items"].append(1)
            return information

        def finish(ctx, result, config):
            config["items"].append(2)
            result.values["answer"] = 99
            ctx.pre_run_data["items"].append(3)

        engine.register_pre_run("prepare", prepare)
        engine.register_post_run("finish", finish)
        self.run_node(workflow, task, engine)
        self.assertEqual(information, {"items": []})
        self.assertEqual(task.executions[0].local["pre_run"], {"items": []})
        self.assertEqual(task.data["answer"], 7)
        self.assertEqual(workflow.nodes[0].pre_run_config, {"items": []})
        self.assertEqual(workflow.nodes[0].post_run_config, {"items": []})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.json"
            workflow.save(path)
            self.assertEqual(WorkflowDefinition.load(path).to_dict(), workflow.to_dict())

    def test_resume_refreshes_context_and_hook_can_be_replaced(self):
        workflow, task, engine = self.setup_node(pre_run="prepare")
        engine.register_pre_run("prepare", lambda ctx, config: {"attempt": 1})
        engine.register("agent", lambda ctx: NodeResult(status="waiting"))
        self.run_node(workflow, task, engine)
        task = Task.from_dict(task.to_dict())
        task.executions[0].status = "ready"
        engine.register_pre_run("prepare", lambda ctx, config: {"attempt": 2})
        engine.register(
            "agent", lambda ctx: NodeResult(values={"answer": ctx.pre_run_data["attempt"]})
        )
        self.run_node(workflow, task, engine)
        self.assertEqual(task.data["answer"], 2)
        self.assertEqual(task.executions[0].local["pre_run"], {"attempt": 2})

    def test_hooks_obey_tool_allowances(self):
        for phase in ("pre_run", "post_run"):
            workflow, task, engine = self.setup_node(**{phase: "tool"}, allowance="restricted")
            engine.allowances.register(
                AllowanceSet("restricted", ["tool:write"], max_tool_risk="low")
            )
            calls = []
            tools = LocalToolModule()
            tools.register(ToolDefinition("write", "high"), lambda *args: calls.append(True))
            engine.attach_tools(tools)
            if phase == "pre_run":
                engine.register_pre_run("tool", lambda ctx, config: ctx.use_tool("write"))
            else:
                engine.register_post_run("tool", lambda ctx, result, config: ctx.use_tool("write"))
            self.assertEqual(self.run_node(workflow, task, engine).status, "failed")
            self.assertEqual(calls, [])
            self.assertEqual(len(task.executions), 1)

    def test_none_pre_return_and_invalid_configuration(self):
        workflow, task, engine = self.setup_node(pre_run="prepare")
        engine.register_pre_run("prepare", lambda ctx, config: None)
        self.assertEqual(self.run_node(workflow, task, engine).status, "completed")
        self.assertEqual(task.executions[0].local["pre_run"], {})
        for phase in ("pre_run", "post_run"):
            for reference in ("", " ", True, 1, {}):
                with self.subTest(phase=phase, reference=reference), self.assertRaises(ValueError):
                    self.setup_node(**{phase: reference})
            with self.assertRaises(ValueError):
                self.setup_node(**{f"{phase}_config": []})
            register = getattr(engine, f"register_{phase}")
            with self.assertRaises(ValueError):
                register("", lambda *args: None)
            with self.assertRaises(TypeError):
                register("invalid", None)


if __name__ == "__main__":
    unittest.main()
