"""The recommended setup: Assistant, safe default allowance, OpenAI-compatible adapter."""

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from workflow_core import (
    MeteredDecision,
    NodeDefinition,
    NodeResult,
    ReActAction,
    TransientModelError,
    WorkflowDefinition,
    create_engine,
    default_allowances,
    openai_compatible,
    use_default_allowance,
)
from workflow_quickstart import Assistant, main


class Scripted:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.payloads = []

    def __call__(self, messages):
        payload = json.loads(messages[-1]["content"])
        self.payloads.append(payload)
        step = self.steps.pop(0)
        return step(payload) if callable(step) else step


def observation(payload):
    return payload["history"][-1]["observation"]


class AssistantTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.workspace = Path(folder.name) / "workspace"
        self.workspace.mkdir()
        (self.workspace / "notes.txt").write_text("Refunds take 5 days.", encoding="utf-8")

    def test_reads_the_workspace_with_the_safe_default(self):
        model = Scripted(
            ReActAction("tool:fs.read_text", {"path": "notes.txt"}),
            lambda p: NodeResult(values={"answer": observation(p)["content"]}),
        )
        reply = Assistant(model, self.workspace).ask("How long do refunds take?")
        self.assertEqual((reply.status, reply.answer), ("completed", "Refunds take 5 days."))
        tools = {tool["name"] for tool in model.payloads[0]["available_tools"]}
        self.assertIn("fs.read_text", tools)
        for unsafe in ("fs.write_text", "fs.delete_file", "http.request", "tools.define"):
            self.assertNotIn(unsafe, tools)
        self.assertEqual(model.payloads[0]["allowance"]["reference"], "safe-default")
        self.assertIn("helpful assistant", model.payloads[0]["node"]["config"]["instruction"])

    def test_writing_needs_allow_writes(self):
        write = ReActAction("tool:fs.write_text", {"path": "out.txt", "content": "x"})
        model = Scripted(write, lambda p: NodeResult(values={"answer": str(observation(p))}))
        reply = Assistant(model, self.workspace).ask("Write a file")
        self.assertIn("error", reply.answer)
        self.assertFalse((self.workspace / "out.txt").exists())

        model = Scripted(write, NodeResult(values={"answer": "written"}))
        reply = Assistant(model, self.workspace, allow_writes=True).ask("Write a file")
        self.assertEqual(reply.answer, "written")
        self.assertEqual((self.workspace / "out.txt").read_text(), "x")

    def test_step_limit_ends_with_an_answer(self):
        loop = ReActAction("tool:fs.list_directory", {"path": "."})
        model = Scripted(loop, loop, NodeResult(values={"answer": "partial"}))
        reply = Assistant(model, self.workspace, max_steps=2).ask("Loop")
        self.assertEqual((reply.status, reply.answer), ("completed", "partial"))
        self.assertTrue(model.payloads[-1]["react_final_round"])

    def test_the_assistant_comes_from_its_preset(self):
        from workflow_quickstart import load_assistant_preset

        preset = load_assistant_preset()
        self.assertEqual((preset["allowance"], preset["model_retry_attempts"]), ("safe-default", 2))
        custom = dict(preset, model_retry_attempts=0, allowance="workspace-editor")
        custom["workflow"]["nodes"][0]["react"]["max_steps"] = 3
        path = self.workspace.parent / "assistant.json"
        path.write_text(json.dumps(custom))
        model = Scripted(NodeResult(values={"answer": "hi"}))
        assistant = Assistant(model, self.workspace, preset=path)
        self.assertEqual(assistant.engine.settings()["model_retry_attempts"], 0)
        self.assertEqual(assistant.workflow.nodes[0].react["max_steps"], 3)
        assistant.ask("hello")
        self.assertEqual(model.payloads[0]["allowance"]["reference"], "workspace-editor")
        path.write_text(json.dumps({**preset, "node": "missing"}))
        with self.assertRaises(ValueError):
            load_assistant_preset(path)
        path.write_text(json.dumps({**preset, "unknown": 1}))
        with self.assertRaises(ValueError):
            load_assistant_preset(path)

    def test_defaults_for_hand_built_workflows(self):
        registry = default_allowances()
        self.assertEqual(set(registry.sets), {"safe-default", "workspace-editor"})
        self.assertEqual(registry.resolve("safe-default").max_tool_risk, "low")
        node = NodeDefinition.from_dict({"id": "a", "type": "agent"})
        workflow = use_default_allowance(WorkflowDefinition("w", "W", [node], [], ["a"]))
        self.assertEqual(workflow.nodes[0].allowance, "safe-default")
        engine = create_engine(self.workspace)
        self.assertEqual(engine.settings()["model_retry_attempts"], 2)
        self.assertIn("fs.read_text", {tool.name for tool in engine.tools.list_tools()})

    def test_terminal_demo(self):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(main(["--demo", "--workspace", str(self.workspace)]), 0)
        self.assertIn("notes.txt", out.getvalue())
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("workflow_model_setup.CONFIG_PATH", self.workspace / "no-config.json"),
            patch("sys.stdin", io.StringIO()),
            patch("sys.stdout", new_callable=io.StringIO) as out,
        ):
            self.assertEqual(main(["--workspace", str(self.workspace)]), 2)  # no model chosen
        self.assertIn("--setup", out.getvalue())


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def reply(message, tokens=None):
    body = {"choices": [{"message": message}]}
    if tokens is not None:
        body["usage"] = {"total_tokens": tokens}
    return Response(json.dumps(body).encode("utf-8"))


def prompt(payload):
    return [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": json.dumps(payload)},
    ]


PAYLOAD = {
    "node": {"config": {"instruction": "Be brief."}},
    "inputs": {"message": "Read notes"},
    "available_tools": [
        {"name": "fs.read_text", "description": "Read.", "input_schema": {"type": "object"}}
    ],
    "history": [
        {
            "action": {"name": "tool:fs.stat", "arguments": {"path": "notes.txt"}},
            "observation": {"exists": True},
        }
    ],
}


class AdapterTests(unittest.TestCase):
    def test_request_and_tool_call(self):
        adapter = openai_compatible("m", "sk-secret", "https://example.com/v1/")
        call = {"function": {"name": "fs_read_text", "arguments": '{"path": "notes.txt"}'}}
        with patch(
            "urllib.request.urlopen",
            return_value=reply({"content": None, "tool_calls": [call]}, 12),
        ) as urlopen:
            decision = adapter(prompt(PAYLOAD))
        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://example.com/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer sk-secret")
        self.assertEqual(body["tools"][0]["function"]["name"], "fs_read_text")
        self.assertEqual(body["messages"][0]["content"], "SYSTEM\n\nBe brief.")
        self.assertEqual(
            [m["role"] for m in body["messages"]], ["system", "user", "assistant", "tool"]
        )
        self.assertEqual(body["messages"][2]["tool_calls"][0]["function"]["name"], "fs_stat")
        self.assertIsInstance(decision, MeteredDecision)
        self.assertEqual(decision.decision, ReActAction("tool:fs.read_text", {"path": "notes.txt"}))

    def test_answer_and_final_round(self):
        adapter = openai_compatible("m")
        payload = {**PAYLOAD, "available_tools": [], "react_final_round": True}
        with patch("urllib.request.urlopen", return_value=reply({"content": " Done. "})) as urlopen:
            decision = adapter(prompt(payload))
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn("tools", body)
        self.assertIn("step limit", body["messages"][-1]["content"])
        self.assertEqual(decision, NodeResult(values={"answer": "Done."}))

    def test_errors(self):
        adapter = openai_compatible("m", "sk-secret")
        busy = urllib.error.HTTPError("u", 429, "busy", {}, io.BytesIO(b"slow down sk-secret"))
        with (
            patch("urllib.request.urlopen", side_effect=busy),
            self.assertRaises(TransientModelError) as caught,
        ):
            adapter(prompt(PAYLOAD))
        self.assertNotIn("sk-secret", str(caught.exception))
        bad = urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"bad request"))
        with (
            patch("urllib.request.urlopen", side_effect=bad),
            self.assertRaises(RuntimeError) as caught,
        ):
            adapter(prompt(PAYLOAD))
        self.assertNotIsInstance(caught.exception, TransientModelError)
        with (
            patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")),
            self.assertRaises(TransientModelError),
        ):
            adapter(prompt(PAYLOAD))

    def test_end_to_end_with_the_assistant(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "notes.txt").write_text("Refunds take 5 days.", encoding="utf-8")
            call = {"function": {"name": "fs_read_text", "arguments": '{"path": "notes.txt"}'}}
            replies = [
                reply({"content": None, "tool_calls": [call]}),
                reply({"content": "Five days."}),
            ]
            with patch("urllib.request.urlopen", side_effect=replies) as urlopen:
                result = Assistant(openai_compatible("m", "k"), folder).ask("Refund time?")
            second = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual((result.status, result.answer), ("completed", "Five days."))
        self.assertIn("Refunds take 5 days.", second["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
