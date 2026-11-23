"""Tool arguments are checked against input_schema; LLM nodes can run without a memory writer."""

import json
import unittest

from workflow_core import (
    LocalToolModule,
    NodeResult,
    ReActAction,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
)
from workflow_schema import validate_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 20},
        "limit": {"type": "integer", "minimum": 1, "maximum": 5},
        "mode": {"enum": ["fast", "full"]},
    },
    "required": ["query"],
    "additionalProperties": False,
}


class SchemaTests(unittest.TestCase):
    def test_accepts_and_rejects(self):
        validate_schema({"query": "x", "limit": 5, "mode": "fast"}, SCHEMA)
        for bad in (
            {},
            {"query": ""},
            {"query": "x" * 21},
            {"query": "x", "limit": 6},
            {"query": "x", "limit": True},
            {"query": "x", "mode": "other"},
            {"query": "x", "extra": 1},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_schema(bad, SCHEMA)


class ToolArgumentTests(unittest.TestCase):
    def test_invalid_arguments_never_reach_the_handler(self):
        calls = []
        tools = LocalToolModule()
        tools.register(
            ToolDefinition("search", "low", input_schema=SCHEMA),
            lambda context, tool, args: calls.append(args) or {"ok": True},
        )
        engine = WorkflowEngine(tools=tools)
        seen = []

        def reason(context, history):
            seen.append(history)
            if not history:
                return ReActAction("tool:search", {"query": "x", "limit": 99})
            if len(history) == 1:
                return ReActAction("tool:search", {"query": "x"})
            return NodeResult()

        engine.register_react("agent", reason)
        workflow = WorkflowDefinition.from_dict(
            {"id": "w", "start": ["a"], "edges": [], "nodes": [{"id": "a", "type": "agent"}]}
        )
        task = Task.create(workflow)
        engine.run_execution(workflow, task, task.executions[0])
        self.assertEqual(calls, [{"query": "x"}])
        error = seen[1][0]["observation"]["error"]
        self.assertEqual(error["type"], "ValueError")
        self.assertIn("arguments.limit", error["message"])


class MemoryWriterTests(unittest.TestCase):
    def test_memory_writer_can_be_disabled(self):
        requests = []

        def model(messages):
            payload = json.loads(messages[-1]["content"])
            requests.append(payload.get("request_type", "decision"))
            return NodeResult()

        engine = WorkflowEngine()
        engine.register_llm("agent", model, memory_writer=False)
        workflow = WorkflowDefinition.from_dict(
            {"id": "w", "start": ["a"], "edges": [], "nodes": [{"id": "a", "type": "agent"}]}
        )
        task = Task.create(workflow)
        result = engine.run_execution(workflow, task, task.executions[0])
        self.assertEqual(result.status, "completed")
        self.assertEqual(requests, ["decision"])
        self.assertEqual(task.active_memory["current"], [])
        self.assertIn("memory.checkpoint", [event["type"] for event in task.events])


if __name__ == "__main__":
    unittest.main()
