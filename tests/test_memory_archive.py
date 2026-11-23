import json
import unittest

from workflow_core import (
    NodeDefinition,
    Task,
    WorkflowDefinition,
    WorkflowEngine,
    append_active_memory,
    read_memory_archive,
)
from workflow_logging import task_memory
from workflow_memory import write_checkpoint


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        workflow = WorkflowDefinition("w", "w", [NodeDefinition("n", "work")], [], ["n", "n"])
        self.task = Task.create(workflow)
        engine = WorkflowEngine()
        self.a, self.b = [
            engine.build_context(workflow, self.task, e) for e in self.task.executions
        ]

    def test_exact_archive_and_prompt_isolation(self):
        original = append_active_memory(self.a, "original-private-note", checkpoint="a")
        sibling = append_active_memory(self.b, "other execution", checkpoint="b")

        def writer(messages):
            self.assertEqual(
                [n["id"] for n in json.loads(messages[1]["content"])["node_notes"]], [original]
            )
            return "summary"

        first = write_checkpoint(self.a, writer, "node_summary", kind="summary", key="end")
        self.assertEqual(write_checkpoint(self.a, None, "node_summary", key="end"), first)
        self.assertEqual([n["id"] for n in self.task.active_memory["current"]], [sibling, first])
        self.assertNotIn("original-private-note", json.dumps(task_memory(self.task)))
        second = append_active_memory(self.a, "updated summary", kind="summary", checkpoint="retry")
        restored = Task.from_dict(json.loads(json.dumps(self.task.to_dict())))
        self.assertEqual(read_memory_archive(restored, summary_id=second)[0]["id"], first)
        detached = read_memory_archive(restored, summary_id=first)
        detached[0]["text"] = "edited"
        self.assertEqual(
            read_memory_archive(restored, summary_id=first)[0]["text"], "original-private-note"
        )
        self.assertEqual(read_memory_archive(restored, execution_id=self.b.execution.id), [])

    def test_failure_leaves_notes_current_and_old_task_migrates(self):
        note = append_active_memory(self.a, "keep", checkpoint="checkpoint")
        with self.assertRaises(ValueError):
            write_checkpoint(self.a, lambda m: "", "node_summary", kind="summary", key="end")
        self.assertEqual(self.task.active_memory["current"][0]["id"], note)
        self.assertEqual(self.task.active_memory["archive"], [])
        raw = self.task.to_dict()
        pending = dict(
            raw["active_memory"]["current"][0], id="other", execution_id=self.b.execution.id
        )
        raw["active_memory"] = raw["active_memory"]["current"] + [
            pending,
            dict(pending, id="summary", kind="summary"),
        ]
        restored = Task.from_dict(raw)
        self.assertEqual([n["id"] for n in restored.active_memory["current"]], [note, "summary"])
        self.assertEqual(read_memory_archive(restored, summary_id="summary")[0]["id"], "other")
