"""Automatic retries of model calls: transient errors only, switchable, journaled."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workflow_core import (
    NodeDefinition,
    NodeResult,
    Task,
    TransientModelError,
    WorkflowDefinition,
    WorkflowEngine,
    is_transient,
)
from workflow_settings import load_settings


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.calls = 0
        self.failures = []
        self.delays = []
        sleeper = patch("workflow_model_retry._sleep", side_effect=self.delays.append)
        sleeper.start()
        self.addCleanup(sleeper.stop)

    def model(self, messages):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return NodeResult(values={"answer": "ok"})

    def run_node(self, overrides=None):
        engine = WorkflowEngine()
        engine.setting_overrides = overrides or {}
        engine.register_llm("agent", self.model, memory_writer=False)
        node = NodeDefinition.from_dict(
            {"id": "a", "type": "agent", "outputs": {"answer": "/answer"}}
        )
        workflow = WorkflowDefinition("w", "W", [node], [], ["a"])
        self.task = Task.create(workflow)
        return engine.run_execution(workflow, self.task, self.task.executions[0])

    def retries(self):
        return [e["details"] for e in self.task.events if e["type"] == "llm.retry"]

    def test_off_by_default(self):
        self.failures = [TransientModelError("timed out")]
        result = self.run_node()
        self.assertEqual(result.status, "failed")
        self.assertIn("timed out", result.error)
        self.assertEqual((self.calls, self.delays), (1, []))

    def test_transient_failures_are_retried_with_backoff(self):
        self.failures = [
            TransientModelError("timed out"),
            TimeoutError("read"),
            ConnectionError("reset"),
        ]
        result = self.run_node({"model_retry_attempts": 3, "model_retry_delay_seconds": 2})
        self.assertEqual(result.status, "completed")
        self.assertEqual((self.calls, self.delays), (4, [2, 4, 8]))
        self.assertEqual(
            [(r["attempt"], r["retries"], r["error_type"]) for r in self.retries()],
            [(1, 3, "TransientModelError"), (2, 3, "TimeoutError"), (3, 3, "ConnectionError")],
        )

    def test_delay_is_capped_and_attempts_run_out(self):
        self.failures = [TransientModelError("busy")] * 5
        result = self.run_node(
            {
                "model_retry_attempts": 3,
                "model_retry_delay_seconds": 10,
                "model_retry_max_delay_seconds": 15,
            }
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual((self.calls, self.delays), (4, [10, 15, 15]))

    def test_other_errors_are_not_retried(self):
        self.failures = [ValueError("bad request")]
        result = self.run_node({"model_retry_attempts": 3})
        self.assertEqual(result.status, "failed")
        self.assertEqual((self.calls, self.retries()), (1, []))

    def test_transient_marker_and_settings_validation(self):
        marked = RuntimeError("overloaded")
        marked.transient = True
        self.assertTrue(is_transient(marked))
        self.assertFalse(is_transient(RuntimeError("x")))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps({"model_retry_attempts": 2}))
            self.assertEqual(load_settings(path)["model_retry_attempts"], 2)
            self.assertEqual(
                load_settings(path, {"model_retry_attempts": 0})["model_retry_attempts"], 0
            )
            with self.assertRaises(ValueError):
                load_settings(path, {"model_retry_attempts": -1})
            with self.assertRaises(ValueError):
                load_settings(path, {"model_retry_delay_seconds": 0})
            with self.assertRaises(ValueError):
                load_settings(path, {"unknown_setting": 1})


if __name__ == "__main__":
    unittest.main()
