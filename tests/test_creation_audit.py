import unittest

from workflow_core import (
    LocalToolModule,
    NodeDefinition,
    NodeResult,
    SkillWriteInterface,
    Task,
    ToolCreationInterface,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
    task_memory,
)


class CreationAuditTests(unittest.TestCase):
    def test_creation_and_reserved_skill_writer(self):
        module = LocalToolModule()
        workflow = WorkflowDefinition("w", "w", [NodeDefinition("n", "work")], [], ["n"])
        task = Task.create(workflow)
        engine = WorkflowEngine(tools=module)

        class Creator(ToolCreationInterface):
            def add_tool(self, ctx, description, request):
                tool = ToolDefinition(request["name"], "low")
                module.register(tool, lambda *args: None)
                module.set_human_check(tool.name)
                return module.get_tool(tool.name)

        def run(ctx):
            with self.assertRaises(NotImplementedError):
                ctx.write_skill(SkillWriteInterface(), "review", {})
            ctx.add_tool(Creator(), ToolDefinition("requested", "low"), {"name": "actual"})
            return NodeResult()

        engine.register("work", run)
        self.assertEqual(
            engine.run_execution(workflow, task, task.executions[0]).status, "completed"
        )
        events = task_memory(task)["events"]
        selected = [
            e
            for e in events
            if e["type"].startswith(("tool.creation.", "skill.write.")) or e["type"] == "tool.added"
        ]
        self.assertEqual(
            [e["type"] for e in selected],
            [
                "skill.write.requested",
                "skill.write.failed",
                "tool.creation.requested",
                "tool.added",
                "tool.creation.completed",
            ],
        )
        self.assertEqual(selected[1]["details"]["request_id"], selected[0]["id"])
        self.assertEqual(selected[-1]["details"]["request_id"], selected[2]["id"])
        self.assertEqual(selected[-1]["details"]["created_tool"]["name"], "actual")
        self.assertNotIn("created_tool", selected[2]["details"])
        self.assertTrue(all(e["execution_id"] == task.executions[0].id for e in selected))
