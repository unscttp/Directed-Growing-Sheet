import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from workflow_core import (
    DEFAULT_PRIVATE_TOOL_CATALOG,
    AllowanceSet,
    BuiltinPrivateToolModule,
    NodeResult,
    ReActAction,
    Task,
    ToolCreationInterface,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
    pending_human_checks,
    resolve_human_check,
)


class PrivateToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.tools = BuiltinPrivateToolModule(self.root)
        self.engine = WorkflowEngine(tools=self.tools)
        self.workflow = WorkflowDefinition.from_dict(
            {
                "id": "private",
                "start": ["agent"],
                "edges": [],
                "nodes": [{"id": "agent", "type": "agent", "outputs": {"answer": "/answer"}}],
            }
        )
        self.task = Task.create(self.workflow)

    def context(self):
        return self.engine.build_context(self.workflow, self.task, self.task.executions[0])

    def policy(self, actions, risk=None):
        self.engine.allowances.register(AllowanceSet("policy", actions, max_tool_risk=risk))
        self.workflow.nodes[0].allowance = "policy"

    def catalog(self):
        return json.loads(DEFAULT_PRIVATE_TOOL_CATALOG.read_text(encoding="utf-8"))

    def reload_catalog(self, raw):
        path = self.root / "catalog.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        self.tools.configure({"workspace_root": str(self.root), "catalog_path": str(path)})

    def test_allowed_descriptions_reach_llm_and_execute(self):
        (self.root / "input.txt").write_text("hello", encoding="utf-8")
        self.policy(["tool:fs.read_text", "tool:fs.write_text"], "low")
        prompts = []

        def llm(messages):
            payload = json.loads(messages[1]["content"])
            prompts.append(payload)
            if payload["history"]:
                return NodeResult(
                    values={"answer": payload["history"][-1]["observation"]["content"]}
                )
            return ReActAction("tool:fs.read_text", {"path": "input.txt"})

        self.engine.register_llm("agent", llm, memory_writer=lambda messages: "Observed checkpoint")
        result = self.engine.run_execution(self.workflow, self.task, self.task.executions[0])
        self.assertEqual(result.status, "completed")
        self.assertEqual(self.task.data["answer"], "hello")
        visible = prompts[0]["available_tools"]
        self.assertEqual([tool["name"] for tool in visible], ["fs.read_text"])
        self.assertTrue(visible[0]["description"])
        self.assertEqual(visible[0]["input_schema"]["required"], ["path"])
        self.assertEqual(prompts[0]["allowance"]["definition"]["max_tool_risk"], "low")
        self.assertNotIn("config", visible[0])

    def test_prompt_is_refiltered_after_policy_change(self):
        self.policy(["tool:fs.read_text"])
        ctx = self.context()
        self.assertEqual(len(ctx.prompt_context["available_tools"]), 1)
        self.engine.allowances.register(AllowanceSet("policy", []))
        self.assertEqual(ctx.prompt_context["available_tools"], [])
        with self.assertRaises(PermissionError):
            ctx.use_tool("fs.read_text", {"path": "any"})

    def test_no_allowance_lists_all_tools(self):
        self.assertEqual(len(self.context().prompt_context["available_tools"]), 20)
        self.policy([])
        self.assertEqual(self.context().prompt_context["available_tools"], [])

    def test_custom_argument_policy_checked_only_at_dispatch(self):
        self.policy(["tool:fs.read_text"])
        checked = []

        def check(ctx, policy, action, arguments):
            checked.append(arguments)
            return arguments["path"] == "public.txt"

        self.engine.allowance_check = check
        ctx = self.context()
        self.assertTrue(ctx.prompt_context["available_tools"][0]["argument_check_required"])
        self.assertEqual(checked, [])
        with self.assertRaises(PermissionError):
            ctx.use_tool("fs.read_text", {"path": "private.txt"})
        self.assertEqual(len(checked), 1)

    def test_six_builtin_tools(self):
        ctx = self.context()
        self.assertFalse(ctx.use_tool("fs.stat", {"path": "missing"})["exists"])
        ctx.use_tool("fs.make_directory", {"path": "nested/deep", "parents": True})
        ctx.use_tool("fs.write_text", {"path": "nested/deep/test.txt", "content": "hello"})
        self.assertEqual(
            ctx.use_tool("fs.read_text", {"path": "nested/deep/test.txt"})["content"], "hello"
        )
        info = ctx.use_tool("fs.stat", {"path": "nested/deep/test.txt"})
        self.assertEqual(info["size_bytes"], 5)
        listing = ctx.use_tool("fs.list_directory", {"path": "nested/deep"})
        self.assertEqual(listing["entries"], [{"name": "test.txt", "type": "file"}])
        ctx.use_tool("fs.write_text", {"path": "data.json", "content": '{"ok": true}'})
        self.assertEqual(ctx.use_tool("fs.read_json", {"path": "data.json"})["data"], {"ok": True})

    def test_no_implicit_overwrite(self):
        ctx = self.context()
        ctx.use_tool("fs.write_text", {"path": "test.txt", "content": "original"})
        with self.assertRaises(FileExistsError):
            ctx.use_tool("fs.write_text", {"path": "test.txt", "content": "new"})
        self.assertEqual((self.root / "test.txt").read_text(), "original")
        ctx.use_tool("fs.write_text", {"path": "test.txt", "content": "new", "overwrite": True})
        self.assertEqual((self.root / "test.txt").read_text(), "new")

    def test_paths_cannot_escape_workspace(self):
        for path in ("../outside.txt", str(self.root / "absolute.txt"), "file.txt:stream", ""):
            with self.subTest(path=path), self.assertRaises((PermissionError, ValueError)):
                self.context().use_tool("fs.write_text", {"path": path, "content": "no"})

    def test_existing_symlink_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as outside:
            try:
                (self.root / "link").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation is unavailable on this host.")
            with self.assertRaises(PermissionError):
                self.context().use_tool("fs.write_text", {"path": "link/out.txt", "content": "no"})
            self.assertFalse((Path(outside) / "out.txt").exists())

    def test_size_limits_and_listing_limit(self):
        (self.root / "large.txt").write_text("12345", encoding="utf-8")
        (self.root / "second.txt").write_text("2", encoding="utf-8")
        self.tools.configure(
            {
                "workspace_root": str(self.root),
                "max_read_bytes": 4,
                "max_write_bytes": 4,
                "max_entries": 1,
            }
        )
        with self.assertRaisesRegex(ValueError, "max_read_bytes"):
            self.context().use_tool("fs.read_text", {"path": "large.txt"})
        with self.assertRaisesRegex(ValueError, "max_write_bytes"):
            self.context().use_tool("fs.write_text", {"path": "new.txt", "content": "12345"})
        self.assertFalse((self.root / "new.txt").exists())
        listing = self.context().use_tool("fs.list_directory", {"path": "."})
        self.assertTrue(listing["truncated"])
        self.assertEqual(len(listing["entries"]), 1)

    def test_bad_arguments_and_invalid_json(self):
        for arguments in ({}, {"path": 3}, {"path": "x", "extra": 1}):
            with self.subTest(arguments=arguments), self.assertRaises((ValueError, TypeError)):
                self.context().use_tool("fs.read_text", arguments)
        with self.assertRaises(TypeError):
            self.context().use_tool("fs.write_text", {"path": "x", "content": "x", "overwrite": 1})
        (self.root / "bad.json").write_text('{"number": NaN}', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.context().use_tool("fs.read_json", {"path": "bad.json"})

    def test_catalog_edits_and_stale_definition(self):
        original = self.tools.get_tool("fs.stat")
        raw = self.catalog()
        raw["tools"][0]["description"] = "Updated description"
        self.reload_catalog(raw)
        self.assertEqual(
            self.context().prompt_context["available_tools"][0]["description"],
            "Updated description",
        )
        with self.assertRaises(RuntimeError):
            self.tools.execute(self.context(), original, {"path": "."})
        copy_of_tool = self.tools.get_tool("fs.stat")
        copy_of_tool.input_schema.clear()
        self.assertTrue(self.tools.get_tool("fs.stat").input_schema)

    def test_invalid_catalog_does_not_replace_working_catalog(self):
        invalid = []
        duplicate = self.catalog()
        duplicate["tools"].append(copy.deepcopy(duplicate["tools"][0]))
        invalid.append(duplicate)
        unknown = self.catalog()
        unknown["tools"][0]["name"] = "execute_generated_code"
        invalid.append(unknown)
        schema = self.catalog()
        schema["tools"][0]["input_schema"]["required"] = []
        invalid.append(schema)
        for raw in invalid:
            with self.assertRaises(ValueError):
                self.reload_catalog(raw)
            self.assertEqual(len(self.tools.list_tools()), 20)

    def test_human_review_visible_but_write_waits(self):
        raw = self.catalog()
        next(tool for tool in raw["tools"] if tool["name"] == "fs.write_text")["human_check"] = True
        self.reload_catalog(raw)
        self.policy(["tool:fs.write_text"], "medium")
        self.assertTrue(self.context().prompt_context["available_tools"][0]["human_check"])
        self.engine.register_react(
            "agent",
            lambda ctx, history: (
                NodeResult()
                if history
                else ReActAction("tool:fs.write_text", {"path": "approved.txt", "content": "ok"})
            ),
        )
        result = self.engine.run_execution(self.workflow, self.task, self.task.executions[0])
        self.assertEqual(result.status, "waiting")
        self.assertFalse((self.root / "approved.txt").exists())
        resolve_human_check(
            self.task, pending_human_checks(self.task)[0]["id"], True, reviewer="tester"
        )
        self.engine.run_execution(self.workflow, self.task, self.task.executions[0])
        self.assertEqual((self.root / "approved.txt").read_text(), "ok")

    def test_legacy_backend_without_discovery_still_works(self):
        class Legacy:
            def get_tool(self, name):
                return ToolDefinition(name, "low")

            def execute(self, context, tool, arguments):
                return 7

        self.engine.attach_tools(Legacy())
        ctx = self.context()
        self.assertFalse(ctx.prompt_context["tool_catalog_available"])
        self.assertEqual(ctx.prompt_context["available_tools"], [])
        self.assertEqual(ctx.use_tool("legacy"), 7)

    def test_creation_is_only_an_interface(self):
        self.assertTrue(inspect.isabstract(ToolCreationInterface))
        with self.assertRaises(TypeError):
            ToolCreationInterface()
        self.assertNotIn("add_tool", [tool.name for tool in self.tools.list_tools()])
        self.assertFalse(hasattr(self.tools, "add_tool"))


if __name__ == "__main__":
    unittest.main()
