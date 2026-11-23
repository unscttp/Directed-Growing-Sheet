"""Choosing the model at setup, and the adapter's Gemini handling."""

import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import workflow_model_setup as setup
from workflow_core import ReActAction, openai_compatible

SERVICES, DEFAULT = setup.load_services()
GEMINI_URL = SERVICES[DEFAULT].base_url


class Answers:
    """Plays the user's answers and records what was shown."""

    def __init__(self, *answers, secret=""):
        self.answers = list(answers)
        self.secret_answer = secret
        self.prompts, self.shown = [], []

    def ask(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0)

    def secret(self, prompt):
        self.prompts.append(prompt)
        return self.secret_answer

    def say(self, text):
        self.shown.append(text)


class SetupTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / ".dgs" / "config.json"
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.checked = []
        self.apis = []

    def run_setup(self, answers, available=(), problems=()):
        problems = list(problems)

        def checker(base_url, key, model, api="chat"):
            self.checked.append((base_url, key, model))
            self.apis.append(api)
            return problems.pop(0) if problems else None

        return setup.setup_model(
            answers.ask,
            answers.secret,
            answers.say,
            path=self.path,
            lister=lambda base_url, key: list(available),
            checker=checker,
        )

    def test_pressing_enter_chooses_gemini_4_hydrogen(self):
        answers = Answers("", "", secret="AIza-test")
        config = self.run_setup(answers, available=["gemini-4-hydrogen", "gemini-4-flash"])
        self.assertEqual(config["model"], "gemini-4-hydrogen")
        self.assertEqual(config["base_url"], GEMINI_URL)
        self.assertIn("Gemini 4 Hydrogen (Google)   (default, press Enter)", answers.shown[1])
        saved = json.loads(self.path.read_text())
        self.assertEqual((saved["model"], saved["api_key"]), ("gemini-4-hydrogen", "AIza-test"))
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(self.checked, [(GEMINI_URL, "AIza-test", "gemini-4-hydrogen")])

    def test_an_unavailable_default_offers_the_family(self):
        answers = Answers("", "2", secret="k")
        available = [
            "gemini-3-pro",
            "gemini-4-flash",
            "gemini-4-hydrogen-preview",
            "text-embedding",
        ]
        config = self.run_setup(answers, available=available)
        self.assertIn("gemini-4-hydrogen is not among the models this key can use.", answers.shown)
        listed = [line.strip() for line in answers.shown if line.startswith("  ") and ")" in line]
        self.assertIn("1) gemini-4-flash", listed)
        self.assertNotIn("text-embedding", " ".join(listed))
        self.assertEqual(config["model"], "gemini-4-hydrogen-preview")

    def test_a_failed_check_lets_the_user_choose_again(self):
        answers = Answers("4", "", "", "deepseek-flash", secret="sk-x")
        config = self.run_setup(answers, problems=["HTTP 404 (not found)"])
        self.assertEqual(config["model"], "deepseek-flash")
        self.assertIn("That did not work: HTTP 404 (not found)", answers.shown)
        self.assertEqual([c[2] for c in self.checked], ["deepseek-v4-pro", "deepseek-flash"])

    def test_the_models_come_from_the_preset(self):
        self.assertEqual(SERVICES[DEFAULT].model, "gemini-4-hydrogen")
        custom = self.path.parent / "models.json"
        custom.parent.mkdir(parents=True, exist_ok=True)
        custom.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "default": "local",
                    "services": [
                        {
                            "id": "local",
                            "label": "Mine",
                            "base_url": "http://localhost:9/v1",
                            "model": "m1",
                            "needs_key": False,
                        }
                    ],
                }
            )
        )
        answers = Answers("", "")
        config = setup.setup_model(
            answers.ask,
            answers.secret,
            answers.say,
            path=self.path,
            lister=lambda b, k: [],
            checker=lambda *a: None,
            services_path=custom,
        )
        self.assertEqual((config["service"], config["model"]), ("local", "m1"))
        self.assertIn("  1) Mine   (default, press Enter)", answers.shown)
        custom.write_text(json.dumps({"format_version": 1, "default": "x", "services": []}))
        with self.assertRaises(ValueError):
            setup.load_services(custom)

    def test_the_menu_offers_the_current_models(self):
        self.assertEqual(
            [s.model for s in SERVICES[:4]],
            ["gemini-4-hydrogen", "claude-opus-5-5", "gpt-6.2-astra", "deepseek-v4-pro"],
        )
        config = self.run_setup(Answers("3", "", secret="sk-o"), available=[])
        self.assertEqual((config["model"], config["api"]), ("gpt-6.2-astra", "responses"))
        self.assertEqual(self.apis, ["responses"])

    def test_a_refused_key_is_asked_for_again(self):
        answers = Answers("", "", "", "")
        keys = iter(["wrong", "right"])
        answers.secret = lambda prompt: next(keys)
        config = self.run_setup(
            answers,
            available=["gemini-4-hydrogen"],
            problems=["HTTP 400: Please pass a valid API key"],
        )
        self.assertEqual(config["api_key"], "right")
        self.assertIn("The key was not accepted. Enter it again? [Y/n] ", answers.prompts)
        self.assertEqual(setup._message('[{"error": {"message": "Bad key"}}]'), "Bad key")

    def test_local_models_need_no_key(self):
        answers = Answers("5", "1")
        config = self.run_setup(answers, available=["llama3.1", "qwen3"])
        self.assertEqual(
            (config["model"], config["base_url"]), ("llama3.1", "http://localhost:11434/v1")
        )
        self.assertNotIn("api_key", config)
        self.assertFalse(any("key" in prompt.lower() for prompt in answers.prompts))

    def test_saved_key_is_kept_and_environment_overrides(self):
        setup.save_config({"base_url": GEMINI_URL, "model": "m", "api_key": "old"}, self.path)
        config = self.run_setup(Answers("", ""), available=["gemini-4-hydrogen"])
        self.assertEqual(config["api_key"], "old")
        with patch.dict(os.environ, {"DGS_MODEL": "override", "DGS_API_KEY": "env-key"}):
            loaded = setup.load_config(self.path)
        self.assertEqual((loaded["model"], loaded["api_key"]), ("override", "env-key"))

    def test_a_service_that_needs_a_key_stops_without_one(self):
        answers = Answers("2")
        self.assertIsNone(self.run_setup(answers))
        self.assertFalse(self.path.exists())


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def prompt(payload):
    return [{"role": "system", "content": "S"}, {"role": "user", "content": json.dumps(payload)}]


class GeminiAdapterTests(unittest.TestCase):
    def test_schemas_signatures_and_tool_names(self):
        adapter = openai_compatible("gemini-4-hydrogen", "k", GEMINI_URL)
        tool = {
            "name": "fs.read_text",
            "description": "Read.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "mode": {"enum": ["a", "b"]}},
                "required": ["path"],
                "additionalProperties": False,
            },
        }
        payload = {"inputs": {"message": "hi"}, "available_tools": [tool], "history": []}
        call = {
            "function": {"name": "fs_read_text", "arguments": '{"path": "a.txt"}'},
            "extra_content": {"google": {"thought_signature": "SIG"}},
        }
        body = {"choices": [{"message": {"content": None, "tool_calls": [call]}}]}
        with patch(
            "urllib.request.urlopen", return_value=Response(json.dumps(body).encode())
        ) as urlopen:
            decision = adapter(prompt(payload))
        sent = json.loads(urlopen.call_args.args[0].data)
        parameters = sent["tools"][0]["function"]["parameters"]
        self.assertNotIn("additionalProperties", parameters)
        self.assertEqual(parameters["properties"]["mode"], {"type": "string", "enum": ["a", "b"]})
        self.assertEqual(
            decision,
            ReActAction("tool:fs.read_text", {"path": "a.txt"}, {"thought_signature": "SIG"}),
        )

        payload["history"] = [
            {
                "action": {"name": "tool:fs.read_text", "arguments": {"path": "a.txt"}},
                "observation": {"content": "x"},
                "model_state": {"thought_signature": "SIG"},
            }
        ]
        answer = {"choices": [{"message": {"content": "done"}}]}
        with patch(
            "urllib.request.urlopen", return_value=Response(json.dumps(answer).encode())
        ) as urlopen:
            adapter(prompt(payload))
        assistant, result = json.loads(urlopen.call_args.args[0].data)["messages"][-2:]
        self.assertEqual(
            assistant["tool_calls"][0]["extra_content"], {"google": {"thought_signature": "SIG"}}
        )
        self.assertEqual(result["name"], "fs_read_text")


class ResponsesAdapterTests(unittest.TestCase):
    def test_reasoning_items_round_trip(self):
        from workflow_core import NodeResult, openai_responses

        adapter = openai_responses("gpt-6.2-astra", "sk-o")
        tool = {"name": "fs.read_text", "description": "Read.", "input_schema": {"type": "object"}}
        payload = {"inputs": {"message": "hi"}, "available_tools": [tool], "history": []}
        reasoning = {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC", "summary": []}
        call = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_9",
            "name": "fs_read_text",
            "arguments": '{"path": "a.txt"}',
        }
        reply = {"output": [reasoning, call], "usage": {"total_tokens": 7}}
        with patch(
            "urllib.request.urlopen", return_value=Response(json.dumps(reply).encode())
        ) as urlopen:
            decision = adapter(prompt(payload))
        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual((body["store"], body["parallel_tool_calls"]), (False, False))
        self.assertEqual(body["include"], ["reasoning.encrypted_content"])
        self.assertEqual(body["tools"][0]["name"], "fs_read_text")
        action = decision.decision
        self.assertEqual((action.name, action.arguments), ("tool:fs.read_text", {"path": "a.txt"}))
        self.assertEqual(action.model_state["items"], [reasoning, call])

        payload["history"] = [
            {
                "action": {"name": "tool:fs.read_text", "arguments": {"path": "a.txt"}},
                "observation": {"content": "x"},
                "model_state": action.model_state,
            }
        ]
        answer = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Done."}]}]
        }
        with patch(
            "urllib.request.urlopen", return_value=Response(json.dumps(answer).encode())
        ) as urlopen:
            final = adapter(prompt(payload))
        items = json.loads(urlopen.call_args.args[0].data)["input"]
        self.assertEqual(items[1:3], [reasoning, call])
        self.assertEqual(
            items[3],
            {"type": "function_call_output", "call_id": "call_9", "output": '{"content": "x"}'},
        )
        self.assertEqual(final, NodeResult(values={"answer": "Done."}))

        payload["history_omitted_older_steps"] = 1  # reasoning no longer matches the input
        with patch(
            "urllib.request.urlopen", return_value=Response(json.dumps(answer).encode())
        ) as urlopen:
            adapter(prompt(payload))
        items = json.loads(urlopen.call_args.args[0].data)["input"]
        self.assertEqual(items[1]["type"], "function_call")
        self.assertNotIn("encrypted_content", json.dumps(items))


class ReasoningContentTests(unittest.TestCase):
    def test_deepseek_reasoning_is_sent_back(self):
        adapter = openai_compatible("deepseek-v4-pro", "k", "https://api.deepseek.com")
        tool = {"name": "fs.stat", "description": "Stat.", "input_schema": {"type": "object"}}
        payload = {"inputs": {"message": "hi"}, "available_tools": [tool], "history": []}
        message = {
            "content": None,
            "reasoning_content": "thinking...",
            "tool_calls": [{"function": {"name": "fs_stat", "arguments": "{}"}}],
        }
        body = {"choices": [{"message": message}]}
        with patch("urllib.request.urlopen", return_value=Response(json.dumps(body).encode())):
            decision = adapter(prompt(payload))
        self.assertEqual(decision.model_state, {"reasoning_content": "thinking..."})
        payload["history"] = [
            {
                "action": {"name": "tool:fs.stat", "arguments": {}},
                "observation": {},
                "model_state": decision.model_state,
            }
        ]
        done = {"choices": [{"message": {"content": "ok"}}]}
        with patch(
            "urllib.request.urlopen", return_value=Response(json.dumps(done).encode())
        ) as urlopen:
            adapter(prompt(payload))
        assistant = json.loads(urlopen.call_args.args[0].data)["messages"][-2]
        self.assertEqual(assistant["reasoning_content"], "thinking...")


if __name__ == "__main__":
    unittest.main()
