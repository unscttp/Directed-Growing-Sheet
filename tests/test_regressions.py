"""Regression tests for problems found in the framework review."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workflow_core import (
    AllowanceRegistry,
    AllowanceSet,
    BuiltinPrivateToolModule,
    CompositeToolModule,
    LocalToolModule,
    NodeResult,
    ReActAction,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
    evaluate_condition,
    pending_human_checks,
    register_skill,
    resolve_human_check,
)


def workflow(nodes, edges=(), start=("a",)):
    return WorkflowDefinition.from_dict(
        {"id": "w", "name": "w", "nodes": list(nodes), "edges": list(edges), "start": list(start)}
    )


class ExecutionStateTests(unittest.TestCase):
    def test_undeclared_output_fails_cleanly(self):
        w = workflow([{"id": "a", "type": "t"}])
        engine = WorkflowEngine()
        engine.register("t", lambda ctx: NodeResult(values={"invented": 1}))
        task = Task.create(w)
        result = engine.run_execution(w, task, task.executions[0])
        self.assertEqual(result.status, "failed")
        self.assertIn("undeclared outputs", result.error)
        self.assertEqual(task.executions[0].status, "failed")
        self.assertEqual(task.status, "failed")
        self.assertEqual(task.data, {})

    def test_handler_returning_wrong_type_fails_cleanly(self):
        w = workflow([{"id": "a", "type": "t"}])
        engine = WorkflowEngine()
        engine.register("t", lambda ctx: {"not": "a NodeResult"})
        task = Task.create(w)
        self.assertEqual(engine.run_execution(w, task, task.executions[0]).status, "failed")
        self.assertEqual(task.status, "failed")

    def test_missing_input_fails_instead_of_staying_running(self):
        w = workflow([{"id": "a", "type": "t", "inputs": {"x": "/missing"}}])
        engine = WorkflowEngine()
        engine.register("t", lambda ctx: NodeResult())
        task = Task.create(w)
        self.assertEqual(engine.run_execution(w, task, task.executions[0]).status, "failed")
        self.assertEqual(task.executions[0].status, "failed")

    def test_condition_on_missing_field_fails_with_guidance_and_can_retry(self):
        w = workflow(
            [{"id": "a", "type": "t"}, {"id": "b", "type": "t"}],
            [{"from": "a", "to": "b", "when": {"eq": ["/flag", True]}}],
        )
        engine = WorkflowEngine()
        engine.register("t", lambda ctx: NodeResult())
        task = Task.create(w)
        result = engine.run_execution(w, task, task.executions[0])
        self.assertIn("guard it with", result.error)
        task.data["flag"] = True
        engine.retry_execution(task, task.executions[0])
        engine.run_execution(w, task, task.executions[0])
        self.assertEqual([e.node_id for e in task.executions], ["a", "b"])

    def test_literal_operand(self):
        data = {"path": "/usr/bin"}
        self.assertTrue(evaluate_condition({"eq": ["/path", {"literal": "/usr/bin"}]}, data))
        with self.assertRaises(KeyError):
            evaluate_condition({"eq": ["/path", "/usr/bin"]}, data)

    def test_rejection_settles_task(self):
        tools = LocalToolModule()
        tools.register(ToolDefinition("risky", "low", human_check=True), lambda *a: {})
        w = workflow([{"id": "a", "type": "t"}])
        engine = WorkflowEngine(tools=tools)
        engine.register("t", lambda ctx: (ctx.use_tool("risky"), NodeResult())[1])
        task = Task.create(w)
        engine.run_execution(w, task, task.executions[0])
        (request,) = pending_human_checks(task)
        resolve_human_check(task, request["id"], False, reviewer="me")
        self.assertEqual(task.status, "failed")
        with self.assertRaises(PermissionError):
            engine.retry_execution(task, task.executions[0])


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.runs = {"A": 0, "B": 0}
        tools = LocalToolModule()
        for name in self.runs:
            tools.register(
                ToolDefinition(name, "low", human_check=True),
                lambda ctx, tool, args: (
                    self.runs.__setitem__(tool.name, self.runs[tool.name] + 1) or {"ran": tool.name}
                ),
            )
        self.engine = WorkflowEngine(tools=tools)
        self.w = workflow([{"id": "a", "type": "t"}])
        self.task = Task.create(self.w)

    def drive(self):
        approvals = 0
        while True:
            result = self.engine.run_execution(self.w, self.task, self.task.executions[0])
            if result.status != "waiting":
                return result, approvals
            (request,) = pending_human_checks(self.task)
            resolve_human_check(self.task, request["id"], True, reviewer="me")
            approvals += 1

    def test_restarted_handler_does_not_repeat_approved_calls(self):
        def handler(ctx):
            first = ctx.use_tool("A")
            second = ctx.use_tool("B")
            return NodeResult(local={"seen": [first, second]})

        self.engine.register("t", handler)
        result, approvals = self.drive()
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.runs, {"A": 1, "B": 1})
        self.assertEqual(approvals, 2)
        types = [e["type"] for e in self.task.events]
        self.assertIn("tool.replayed", types)
        self.assertNotIn("action_journal", self.task.executions[0].local)

    def test_divergent_restart_executes_normally(self):
        counted = []
        self.engine.tools.register(
            ToolDefinition("count", "low"), lambda ctx, tool, args: counted.append(args) or {}
        )
        attempts = []

        def handler(ctx):
            attempts.append(1)
            ctx.use_tool("count", {"attempt": len(attempts)})  # differs on every run
            ctx.use_tool("B")
            return NodeResult()

        self.engine.register("t", handler)
        result, approvals = self.drive()
        self.assertEqual(result.status, "completed")
        self.assertEqual(counted, [{"attempt": 1}, {"attempt": 2}])
        self.assertIn("replay.diverged", [e["type"] for e in self.task.events])


class PromptBoundTests(unittest.TestCase):
    def test_prompt_size_stops_growing(self):
        tools = LocalToolModule()
        tools.register(ToolDefinition("read", "low"), lambda *a: {"content": "x" * 5000})
        w = workflow([{"id": "a", "type": "agent", "react": {"max_steps": 200}}])
        engine = WorkflowEngine(tools=tools)
        sizes = []

        def llm(messages):
            sizes.append(sum(len(m["content"]) for m in messages))
            return ReActAction("tool:read") if len(sizes) < 120 else NodeResult()

        engine.register_llm("agent", llm, memory_writer=lambda m: "note")
        task = Task.create(w)
        engine.run_execution(w, task, task.executions[0])
        self.assertLess(max(sizes), 120_000)
        self.assertLess(sizes[-1], sizes[59] * 1.2)

    def test_snapshot_does_not_keep_stale_streams(self):
        w = workflow([{"id": "a", "type": "t"}])
        task = Task.create(w)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "task.json"
            for index in range(5):
                task.metadata["i"] = index
                from workflow_core import record_event

                record_event(task, "app.note", {"i": index})
                task.save(path)
            streams = list((Path(folder) / "task.json.memory").rglob("*.jsonl"))
            self.assertEqual(len(streams), 3)
            self.assertEqual(len(Task.load(path).events), len(task.events))


class GeneratedGraphHostCodeTests(unittest.TestCase):
    def test_graph_cannot_reference_hooks_without_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / "settings.json"
            settings.write_text(json.dumps({"graph_creation_human_check": False}))
            target = Path(folder) / "written.txt"
            registry = AllowanceRegistry()
            registry.register(AllowanceSet("planner", ["tool:graph.create"], max_tool_risk="high"))
            engine = WorkflowEngine(registry, memory_settings_path=settings)
            engine.register_post_run(
                "write", lambda ctx, result, config: Path(config["path"]).write_text("x")
            )
            engine.register("noop", lambda ctx: NodeResult(), generated_graphs=True)
            graph = {
                "explanation": "helper",
                "start": ["x"],
                "edges": [],
                "nodes": [
                    {
                        "id": "x",
                        "type": "noop",
                        "post_run": "write",
                        "post_run_config": {"path": str(target)},
                    }
                ],
            }
            history_seen = []

            def plan(ctx, history):
                history_seen.extend(history)
                return NodeResult() if history else ReActAction("tool:graph.create", graph)

            engine.register_react("agent", plan)
            w = workflow(
                [{"id": "a", "type": "agent", "allowance": "planner", "allow_create_graph": True}]
            )
            task = Task.create(w)
            engine.run_execution(w, task, task.executions[0])
            self.assertFalse(target.exists())
            self.assertIn("post_run 'write'", history_seen[0]["observation"]["error"]["message"])

            engine.register_post_run(
                "write",
                lambda ctx, result, config: Path(config["path"]).write_text("x"),
                generated_graphs=True,
            )
            task = Task.create(w)
            engine.run_execution(w, task, task.executions[0])
            self.assertTrue(target.exists())


class ToolModuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "ws").mkdir()

    def test_skills_and_builtin_tools_together(self):
        package = self.root / "demo"
        package.mkdir()
        (package / "SKILL.md").write_text("---\nname: demo\ndescription: d\n---\nbody\n")
        skills = LocalToolModule()
        register_skill(skills, package)
        builtin = BuiltinPrivateToolModule(self.root / "ws")
        tools = CompositeToolModule(builtin, skills)
        names = {tool.name for tool in tools.list_tools()}
        self.assertIn("fs.read_text", names)
        self.assertIn("skill.demo.invoke", names)
        self.assertIn("demo", tools.skill_packages)
        self.assertIs(tools.safety_policy, builtin.safety_policy)
        engine = WorkflowEngine(tools=tools)
        w = workflow([{"id": "a", "type": "t", "outputs": {"r": "/r"}}])

        def handler(ctx):
            ctx.use_tool("fs.write_text", {"path": "f.txt", "content": "hi"})
            invoke = ctx.use_tool("skill.demo.invoke", {})["invoke_id"]
            ctx.use_tool("skill.demo.finish", {"invoke_id": invoke})
            return NodeResult(values={"r": ctx.use_tool("fs.read_text", {"path": "f.txt"})})

        engine.register("t", handler)
        task = Task.create(w)
        self.assertEqual(engine.run_execution(w, task, task.executions[0]).status, "completed")
        self.assertEqual(task.data["r"]["content"], "hi")

    def test_duplicate_tool_names_rejected(self):
        one, two = LocalToolModule(), LocalToolModule()
        for module in (one, two):
            module.register(ToolDefinition("same", "low"), lambda *a: {})
        with self.assertRaises(ValueError):
            CompositeToolModule(one, two)

    def test_template_directory_check_ignores_case(self):
        tools = BuiltinPrivateToolModule(self.root / "ws")
        for raw in (".dgs-tools/definitions.json", ".DGS-Tools/definitions.json"):
            with self.subTest(raw=raw), self.assertRaises(PermissionError):
                tools._path(raw)

    def test_http_credentials_are_host_managed(self):
        tools = BuiltinPrivateToolModule(
            self.root / "ws",
            http_allowed_origins=["https://api.example.com"],
            http_credentials={"https://api.example.com": {"Authorization": "Bearer s3cret"}},
        )
        engine = WorkflowEngine(tools=tools)
        w = workflow([{"id": "a", "type": "t"}])
        args = {"url": "https://api.example.com/x", "method": "GET"}

        def handler(ctx):
            with self.assertRaises(PermissionError):
                ctx.use_tool("http.request", {**args, "headers": {"authorization": "Bearer x"}})
            ctx.use_tool("http.request", args)
            return NodeResult()

        engine.register("t", handler)
        addresses = [(2, 1, 6, "", ("93.184.216.34", 443))]
        with (
            patch("workflow_http_tools.socket.getaddrinfo", return_value=addresses),
            patch("workflow_http_tools.socket.socket"),
            patch("workflow_http_tools.ssl.create_default_context"),
            patch("workflow_http_tools.http.client.HTTPConnection") as connection,
        ):
            response = connection.return_value.getresponse.return_value
            response.read.return_value = b"{}"
            response.status, response.getheaders.return_value = 200, []
            task = Task.create(w)
            self.assertEqual(engine.run_execution(w, task, task.executions[0]).status, "completed")
            sent = connection.return_value.request.call_args.kwargs["headers"]
        self.assertEqual(sent["Authorization"], "Bearer s3cret")
        self.assertNotIn("s3cret", json.dumps(task.to_dict()))
        self.assertNotIn("s3cret", repr(tools._http))

    def test_credentials_require_allowed_origin(self):
        with self.assertRaises(ValueError):
            BuiltinPrivateToolModule(
                self.root / "ws",
                http_credentials={"https://other.example.com": {"Authorization": "x"}},
            )


if __name__ == "__main__":
    unittest.main()
