import json
import tempfile
import unittest
from pathlib import Path

from workflow_core import (
    HumanCheckRequired,
    LocalToolModule,
    NodeDefinition,
    SkillAgentInterface,
    SkillGraphInterface,
    Task,
    WorkflowDefinition,
    WorkflowEngine,
    pending_human_checks,
    register_skill,
    resolve_human_check,
)


class SkillTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "review"
        self.root.mkdir()
        (self.root / "SKILL.md").write_text(
            "---\nname: review\ndescription: >\n  Review a document\n  carefully.\n"
            "metadata: {author: test}\nallowed-tools: Everything\n---\n"
            "Follow references/guide.md",
            encoding="utf8",
        )
        (self.root / "references").mkdir()
        (self.root / "references/guide.md").write_text("guide", encoding="utf8")
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts/check.py"
        self.script.write_text("import sys; print(sys.argv[1])", encoding="utf8")
        module = LocalToolModule()
        register_skill(module, self.root, scripts=["scripts/check.py"])
        self.engine = WorkflowEngine(tools=module)
        self.workflow = WorkflowDefinition("w", "w", [NodeDefinition("n", "work")], [], ["n"])
        self.task = Task.create(self.workflow)

    def call(self, op, **args):
        ctx = self.engine.build_context(self.workflow, self.task, self.task.executions[0])
        return ctx.use_tool("skill.review." + op, args)

    def test_logged_nested_lifecycle_survives_restore(self):
        root = self.call("invoke")
        self.assertIn("carefully.", root["metadata"]["description"])
        self.assertIn("references/guide.md", root["instructions"])
        self.task = Task.from_dict(json.loads(json.dumps(self.task.to_dict())))
        child = self.call("invoke", parent_invoke_id=root["invoke_id"])["invoke_id"]
        self.assertEqual(
            self.call("read", invoke_id=child, path="references/guide.md")["content"], "guide"
        )
        with self.assertRaises(ValueError):
            self.call("finish", invoke_id=root["invoke_id"])
        self.call("finish", invoke_id=child)
        self.call("finish", invoke_id=root["invoke_id"])
        with self.assertRaises(ValueError):
            self.call("invoke", parent_invoke_id=child)
        event = next(e for e in self.task.events if e["type"] == "skill.read.completed")
        self.assertEqual(event["details"]["parent_invoke_id"], root["invoke_id"])
        self.assertEqual(event["details"]["invoke_id"], child)

    def test_script_approval_and_result(self):
        invocation = self.call("invoke")["invoke_id"]
        args = {"invoke_id": invocation, "path": "scripts/check.py", "argv": ["hello"]}
        with self.assertRaises(HumanCheckRequired):
            self.call("script", **args)
        self.assertNotIn("skill.script.started", [e["type"] for e in self.task.events])
        self.task.executions[0].status = "waiting"
        resolve_human_check(
            self.task, pending_human_checks(self.task)[0]["id"], True, reviewer="tester"
        )
        self.assertEqual(self.call("script", **args)["stdout"].strip(), "hello")
        self.assertIn("skill.script.exited", [e["type"] for e in self.task.events])

    def test_paths_changed_scripts_and_failed_load(self):
        invocation = self.call("invoke")["invoke_id"]
        with self.assertRaises(ValueError):
            self.call("read", invoke_id=invocation, path="../outside")
        args = {"invoke_id": invocation, "path": "scripts/check.py"}
        with self.assertRaises(HumanCheckRequired):
            self.call("script", **args)
        self.task.executions[0].status = "waiting"
        resolve_human_check(
            self.task, pending_human_checks(self.task)[0]["id"], True, reviewer="tester"
        )
        self.script.write_text("raise SystemExit(3)")
        with self.assertRaises(PermissionError):
            self.call("script", **args)
        (self.root / "SKILL.md").write_text("broken")
        with self.assertRaises(ValueError):
            self.call("invoke")
        self.assertIn("skill.load.failed", [e["type"] for e in self.task.events])

    def test_reserved_interfaces_do_not_execute(self):
        with self.assertRaises(NotImplementedError):
            SkillAgentInterface().invoke_agent(None, "id", {})
        # The graph hook is implemented; it needs a node context with an engine.
        with self.assertRaises(AttributeError):
            SkillGraphInterface().execute_graph(None, "id", "graph")
