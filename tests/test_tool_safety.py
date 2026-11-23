import json
import tempfile
import unittest
from pathlib import Path

from workflow_core import (
    BuiltinPrivateToolModule,
    HumanCheckRequired,
    NodeResult,
    ReActAction,
    Task,
    WorkflowDefinition,
    WorkflowEngine,
    pending_human_checks,
    resolve_human_check,
)


class ToolSafetyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.events = []
        self.tools = BuiltinPrivateToolModule(self.root, on_security_event=self.events.append)
        self.engine = WorkflowEngine(tools=self.tools)
        self.workflow = WorkflowDefinition.from_dict(
            {
                "id": "safety",
                "start": ["agent"],
                "edges": [],
                "nodes": [{"id": "agent", "type": "agent"}],
            }
        )
        self.task = Task.create(self.workflow)
        self.file = self.root / "file.txt"
        self.file.write_text("keep", encoding="utf-8")

    def context(self):
        return self.engine.build_context(self.workflow, self.task, self.task.executions[0])

    def configure(self, **settings):
        path = self.root / "safety.json"
        path.write_text(json.dumps(settings), encoding="utf-8")
        self.tools.configure({"workspace_root": str(self.root), "safety_config_path": str(path)})

    def run_delete(self):
        self.engine.register_react(
            "agent",
            lambda ctx, history: (
                NodeResult()
                if history
                else ReActAction("tool:fs.delete_file", {"path": "file.txt"})
            ),
        )
        return self.engine.run_execution(self.workflow, self.task, self.task.executions[0])

    def test_delete_pauses_and_resumes_once_after_persisted_approval(self):
        self.assertTrue(self.tools.get_tool("fs.delete_file").human_check)
        self.assertEqual(self.run_delete().status, "waiting")
        self.assertTrue(self.file.exists())
        self.assertEqual(self.events[0]["kind"], "delete_requires_human")
        self.task = Task.from_dict(json.loads(json.dumps(self.task.to_dict())))
        request = pending_human_checks(self.task)[0]
        self.assertEqual(request["id"], self.events[0]["request_id"])
        resolve_human_check(self.task, request["id"], True, reviewer="human")
        self.assertEqual(self.run_delete().status, "completed")
        self.assertFalse(self.file.exists())
        self.assertEqual(len(self.task.executions[0].human_checks), 1)
        self.assertEqual(len(self.task.executions[0].local["tool_security_events"]), 1)

    def test_rejection_preserves_file(self):
        self.run_delete()
        resolve_human_check(
            self.task, pending_human_checks(self.task)[0]["id"], False, reviewer="human"
        )
        self.assertEqual(self.task.executions[0].status, "failed")
        self.assertTrue(self.file.exists())

    def test_changed_file_needs_new_approval(self):
        self.run_delete()
        request = pending_human_checks(self.task)[0]
        resolve_human_check(self.task, request["id"], True, reviewer="human")
        self.file.write_text("different content", encoding="utf-8")
        self.assertEqual(self.run_delete().status, "waiting")
        self.assertNotEqual(pending_human_checks(self.task)[0]["id"], request["id"])
        self.assertTrue(self.file.exists())

    def test_catalog_human_check_is_combined_with_delete_lock(self):
        from workflow_core import DEFAULT_PRIVATE_TOOL_CATALOG

        raw = json.loads(DEFAULT_PRIVATE_TOOL_CATALOG.read_text(encoding="utf-8"))
        next(t for t in raw["tools"] if t["name"] == "fs.delete_file")["human_check"] = True
        path = self.root / "catalog.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        self.tools.configure({"workspace_root": str(self.root), "catalog_path": str(path)})
        self.run_delete()
        resolve_human_check(
            self.task, pending_human_checks(self.task)[0]["id"], True, reviewer="human"
        )
        self.assertEqual(self.run_delete().status, "completed")
        self.assertEqual(len(self.task.executions[0].human_checks), 1)

    def test_direct_execution_cannot_bypass_delete_approval(self):
        with self.assertRaises(PermissionError):
            self.tools.execute(
                self.context(), self.tools.get_tool("fs.delete_file"), {"path": "file.txt"}
            )
        self.assertTrue(self.file.exists())

    def test_disable_delete_lock(self):
        self.configure(delete_requires_human=False)
        self.context().use_tool("fs.delete_file", {"path": "file.txt"})
        self.assertFalse(self.file.exists())
        self.assertEqual(self.events, [])

    def test_system_edits_blocked_before_human_gate(self):
        self.configure(additional_protected_paths=[str(self.file)])
        for tool, args in (
            ("fs.write_text", {"path": "file.txt", "content": "bad", "overwrite": True}),
            ("fs.delete_file", {"path": "file.txt"}),
        ):
            with self.assertRaisesRegex(PermissionError, "System-file"):
                self.context().use_tool(tool, args)
        self.assertEqual(self.file.read_text(), "keep")
        self.assertEqual(pending_human_checks(self.task), [])
        self.assertEqual([e["kind"] for e in self.events], ["system_edit_blocked"] * 2)
        self.assertEqual(
            self.context().use_tool("fs.read_text", {"path": "file.txt"})["content"], "keep"
        )

    def test_disable_system_lock_keeps_delete_lock(self):
        self.configure(block_system_edits=False, additional_protected_paths=[str(self.file)])
        self.context().use_tool(
            "fs.write_text", {"path": "file.txt", "content": "ok", "overwrite": True}
        )
        self.assertEqual(self.file.read_text(), "ok")
        with self.assertRaises(HumanCheckRequired):
            self.context().use_tool("fs.delete_file", {"path": "file.txt"})

    def test_protected_absolute_path_warns_and_boundary_is_exact(self):
        self.configure(additional_protected_paths=[str(self.root / "system")])
        with self.assertRaisesRegex(PermissionError, "System-file"):
            self.context().use_tool(
                "fs.write_text", {"path": str(self.root / "system" / "x"), "content": "x"}
            )
        self.context().use_tool("fs.make_directory", {"path": "system-other"})
        self.assertTrue((self.root / "system-other").is_dir())

    def test_callback_failure_still_blocks(self):
        self.configure(additional_protected_paths=[str(self.file)])

        def broken(event):
            raise RuntimeError("UI unavailable")

        self.tools.on_security_event = broken
        with self.assertRaises(RuntimeError):
            self.context().use_tool(
                "fs.write_text", {"path": "file.txt", "content": "bad", "overwrite": True}
            )
        self.assertEqual(self.file.read_text(), "keep")
        self.assertEqual(len(self.task.executions[0].local["tool_security_events"]), 1)

    def test_invalid_policy_reload_is_atomic(self):
        for settings in (
            {"delete_requires_human": "false"},
            {"block_system_edits": 0},
            {"additional_protected_paths": ["relative"]},
            {"typo": False},
        ):
            with self.assertRaises(ValueError):
                self.configure(**settings)
            self.assertTrue(self.tools.get_tool("fs.delete_file").human_check)

    def test_no_recursive_deletion_and_workspace_guard_remains(self):
        self.configure(delete_requires_human=False, block_system_edits=False)
        with self.assertRaises(ValueError):
            self.context().use_tool("fs.delete_file", {"path": "."})
        with self.assertRaises(PermissionError):
            self.context().use_tool("fs.write_text", {"path": "../outside", "content": "bad"})


if __name__ == "__main__":
    unittest.main()
