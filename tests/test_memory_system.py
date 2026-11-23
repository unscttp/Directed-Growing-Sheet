import json
import tempfile
import unittest
from pathlib import Path

from workflow_core import (
    MeteredDecision,
    NodeDefinition,
    NodeResult,
    ReActAction,
    Task,
    WorkflowDefinition,
    WorkflowEngine,
)
from workflow_memory import write_checkpoint


class MemorySystemTests(unittest.TestCase):
    def setUp(self):
        self.workflow = WorkflowDefinition("w", "W", [NodeDefinition("n", "agent")], [], ["n"])
        self.task = Task.create(self.workflow)

    def test_live_limits_citations_and_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory) / "settings.json"
            settings.write_text('{"react_rounds": 9, "token_limit": 12}')
            engine = WorkflowEngine(memory_settings_path=settings)
            decisions = iter([ReActAction("change"), NodeResult()])

            def change(ctx, args):
                settings.write_text('{"react_rounds": 2, "token_limit": 10}')

            def writer(messages):
                self.assertIn("takes precedence", messages[0]["content"])
                checkpoint = json.loads(messages[1]["content"])["checkpoint"]
                return {"text": checkpoint["name"], "references": [checkpoint["event_id"]]}

            engine.register_action("change", change)
            engine.register_llm(
                "agent", lambda messages: MeteredDecision(next(decisions), 5), memory_writer=writer
            )
            result = engine.run_execution(self.workflow, self.task, self.task.executions[0])
            self.assertEqual(result.status, "completed", result.error)
            (summary,) = self.task.active_memory["current"]
            notes = self.task.active_memory["archive"]
            self.assertEqual(
                {n["checkpoint"] for n in notes}, {"node_start", "react_rounds", "token_limit"}
            )
            self.assertEqual(summary["archived_ids"], [n["id"] for n in notes])
            ids = {e["id"] for e in self.task.events}
            for note in [*notes, summary]:
                self.assertTrue(note["references"])
                self.assertTrue(set(note["references"]) <= ids)

            path = Path(directory) / "task.json"
            self.task.save(path)
            manifest = json.loads(path.read_text())["memory_streams"]
            archive = (
                path.parent / "task.json.memory" / "archive" / (manifest["archive"] + ".jsonl")
            )
            self.assertEqual(
                [json.loads(line) for line in archive.read_bytes().splitlines()], notes
            )
            self.assertEqual(Task.load(path).to_dict(), self.task.to_dict())
            archive.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "Damaged archive"):
                Task.load(path)

    def test_reject_fabricated_citation_before_publishing(self):
        engine = WorkflowEngine()
        ctx = engine.build_context(self.workflow, self.task, self.task.executions[0])
        with self.assertRaisesRegex(ValueError, "unknown objective event"):
            write_checkpoint(ctx, lambda _: {"text": "note", "references": ["invented"]}, "review")
        self.assertEqual(self.task.active_memory["current"], [])
