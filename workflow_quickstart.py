"""The recommended way to start: a ready-made assistant with safe defaults.

    from workflow_quickstart import Assistant, openai_compatible

    model = openai_compatible(
        "gemini-4-hydrogen",
        api_key="...",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )
    assistant = Assistant(model, workspace="workspace")
    reply = assistant.ask("Summarise notes.txt")
    print(reply.answer)

``Assistant`` is one LLM node on an engine set up with:

* the built-in toolbox (``BuiltinPrivateToolModule``), confined to the
  ``workspace`` folder;
* the ``safe-default`` allowance (``presets/allowances/safe-default.json``): reading files
  in the workspace and working with text, JSON and CSV; nothing is written,
  deleted or sent over the network. ``allow_writes=True`` uses
  ``workspace-editor`` instead, which may also write files and create folders;
* automatic retries of model calls that time out (``model_retry_attempts``);
* an answer instead of a failure when the step limit is reached
  (``react.on_step_limit = "answer"``).

``openai_compatible`` is a small model adapter for any API that speaks the
OpenAI Chat Completions format (Gemini through Google's compatible endpoint,
DeepSeek, OpenAI, local servers such as Ollama or LM Studio, and others). It
uses only the standard library. For other providers, write an adapter as
described in README.md ("Connecting a model").

``create_engine`` and ``use_default_allowance`` give hand-built workflows the
same defaults. ``python install.py`` asks which model to use and saves the
choice (``workflow_model_setup``); ``python quickstart.py`` then opens a chat in
the terminal.
"""

from __future__ import annotations

import copy
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from workflow_allowances import AllowanceRegistry
from workflow_approvals import pending_human_checks, resolve_human_check
from workflow_definitions import WorkflowDefinition
from workflow_engine import WorkflowEngine
from workflow_executions import Task
from workflow_model_retry import TransientModelError
from workflow_nodes import NodeResult
from workflow_private_tools import BuiltinPrivateToolModule
from workflow_prompts import MeteredDecision
from workflow_react import ReActAction
from workflow_schema import validate_document
from workflow_schema_compat import gemini_parameters
from workflow_types import JsonDict

# Every default of the recommended setup is JSON in presets/:
#   allowances/*.json  the allowance sets (safe-default, workspace-editor)
#   assistant.json     the assistant's workflow, permissions and retries
#   models.json        the models the one-step setup offers (workflow_model_setup)
PRESETS = Path(__file__).with_name("presets")
ALLOWANCE_PRESETS = PRESETS / "allowances"
ASSISTANT_PRESET = PRESETS / "assistant.json"
SAFE_DEFAULT = "safe-default"
FINAL_ROUND_NOTE = (
    "The step limit is reached and no tools are available any more. Answer now with "
    "what you have found, and say what is missing."
)


# -- defaults for any engine ---------------------------------------------------


def default_allowances(folder: str | Path | None = None) -> AllowanceRegistry:
    """A registry with every allowance in ``presets/allowances`` (or ``folder``)."""
    registry = AllowanceRegistry()
    for path in sorted(Path(folder or ALLOWANCE_PRESETS).glob("*.json")):
        registry.load(path)
    return registry


def load_assistant_preset(path: str | Path | None = None) -> JsonDict:
    """The assistant preset (``presets/assistant.json``), validated."""
    preset = json.loads(Path(path or ASSISTANT_PRESET).read_text("utf-8"))
    validate_document(preset, "assistant_preset")
    if not any(node["id"] == preset["node"] for node in preset["workflow"]["nodes"]):
        raise ValueError(f"The preset's node {preset['node']!r} is not in its workflow.")
    return preset


def use_default_allowance(workflow: WorkflowDefinition, allowance: str = SAFE_DEFAULT):
    """Give every node without an allowance the safe default (nodes without one are
    otherwise unrestricted). Returns the workflow."""
    for node in workflow.nodes:
        if node.allowance is None:
            node.allowance = allowance
    workflow.assert_valid()
    return workflow


def create_engine(
    workspace: str | Path = "workspace",
    *,
    retries: int | None = None,
    memory_settings_path: str | Path | None = None,
) -> WorkflowEngine:
    """An engine with the built-in toolbox in ``workspace`` and the bundled presets.

    Register your node types on it and pass your workflow through
    ``use_default_allowance`` so its nodes get the safe default.
    """
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    engine = WorkflowEngine(
        default_allowances(),
        tools=BuiltinPrivateToolModule(root),
        memory_settings_path=memory_settings_path,
    )
    if retries is None:
        retries = load_assistant_preset()["model_retry_attempts"]
    engine.setting_overrides = {"model_retry_attempts": retries}
    return engine


# -- the assistant -------------------------------------------------------------


@dataclass
class Reply:
    """The outcome of ``Assistant.ask`` or ``Assistant.decide``."""

    status: str  # completed, waiting (for an approval) or failed
    answer: str | None
    error: str | None
    pending: list[JsonDict] = field(default_factory=list)  # approvals to decide
    task: Task | None = None


class Assistant:
    """One agent with safe defaults, defined by ``presets/assistant.json``.

    The preset holds the workflow (one LLM node with its instructions, step
    limit and allowance), the allowance used with ``allow_writes``, the
    automatic retries and the default workspace. ``preset=`` loads another
    file; the keyword arguments override single values. Each ``ask`` runs one
    task.
    """

    def __init__(
        self,
        model,
        workspace: str | Path | None = None,
        *,
        preset: str | Path | None = None,
        instructions: str | None = None,
        allow_writes: bool = False,
        max_steps: int | None = None,
        retries: int | None = None,
    ) -> None:
        if not callable(model):
            raise TypeError("model must be a model adapter, for example openai_compatible(...).")
        settings = load_assistant_preset(preset)
        document = copy.deepcopy(settings["workflow"])
        node = next(n for n in document["nodes"] if n["id"] == settings["node"])
        node["allowance"] = settings["allowance_with_writes" if allow_writes else "allowance"]
        if instructions is not None:
            node.setdefault("config", {})["instruction"] = instructions
        if max_steps is not None:
            node.setdefault("react", {})["max_steps"] = max_steps
        self.workflow = WorkflowDefinition.from_dict(document)
        self.workflow.assert_valid()
        self.workspace = Path(workspace or settings["workspace"])
        self.engine = create_engine(
            self.workspace,
            retries=settings["model_retry_attempts"] if retries is None else retries,
        )
        self.engine.register_llm(node["type"], model, memory_writer=False)

    def ask(self, message: str) -> Reply:
        """Run a new task for ``message``."""
        task = Task.create(self.workflow, {"message": message})
        return self.resume(task)

    def resume(self, task: Task) -> Reply:
        """Continue a task, for example after loading it with ``Task.load``."""
        self.engine.task_pool.add(task)
        while ready := self.engine.task_pool.ready():
            for ready_task, execution in ready:
                self.engine.run_execution(self.workflow, ready_task, execution)
        pending = [
            {"id": r["id"], "kind": r["details"].get("kind"), "details": r["details"]}
            for r in pending_human_checks(task)
        ]
        errors = [e.error for e in task.executions if e.status == "failed" and e.error]
        if pending:
            status = "waiting"
        else:
            status = "completed" if task.status == "completed" else "failed"
        return Reply(
            status=status,
            answer=task.data.get("answer"),
            error=errors[-1] if errors and status == "failed" else None,
            pending=pending,
            task=task,
        )

    def decide(
        self, reply: Reply, request_id: str, approved: bool, *, reviewer: str = "user"
    ) -> Reply:
        """Approve or reject a pending request of ``reply`` and continue."""
        resolve_human_check(reply.task, request_id, approved, reviewer=reviewer)
        return self.resume(reply.task)


# -- an adapter for OpenAI-compatible APIs ----------------------------------------


def _function_name(tool_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)[:64]


def _system_text(messages: list[JsonDict], payload: JsonDict) -> str:
    instruction = ((payload.get("node") or {}).get("config") or {}).get("instruction") or ""
    return messages[0]["content"] + ("\n\n" + instruction if instruction else "")


def _request_text(payload: JsonDict) -> str:
    inputs = payload.get("inputs") or {}
    request = inputs.get("message")
    return request if isinstance(request, str) else json.dumps(inputs, ensure_ascii=False)


def _chat_messages(
    messages: list[JsonDict], payload: JsonDict, gemini: bool = False
) -> list[JsonDict]:
    chat = [
        {"role": "system", "content": _system_text(messages, payload)},
        {"role": "user", "content": _request_text(payload)},
    ]
    for number, entry in enumerate(payload.get("history") or []):
        observation = json.dumps(entry.get("observation"), ensure_ascii=False)
        if "action" not in entry:  # a final result that was rejected
            chat.append({"role": "user", "content": f"Your result was not accepted: {observation}"})
            continue
        call_id = f"call_{number}"
        name = entry["action"]["name"].removeprefix("tool:")
        arguments = json.dumps(entry["action"].get("arguments") or {}, ensure_ascii=False)
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {"name": _function_name(name), "arguments": arguments},
        }
        signature = (entry.get("model_state") or {}).get("thought_signature")
        if signature:  # Gemini refuses a tool-call history without its signatures
            tool_call["extra_content"] = {"google": {"thought_signature": signature}}
        assistant = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        if (entry.get("model_state") or {}).get("reasoning_content"):
            # Thinking models (DeepSeek) expect their reasoning back with the call.
            assistant["reasoning_content"] = entry["model_state"]["reasoning_content"]
        chat.append(assistant)
        result = {"role": "tool", "tool_call_id": call_id, "content": observation}
        if gemini:
            result["name"] = _function_name(name)
        chat.append(result)
    if payload.get("react_final_round"):
        chat.append({"role": "user", "content": FINAL_ROUND_NOTE})
    return chat


def is_gemini(base_url: str) -> bool:
    return "generativelanguage.googleapis.com" in base_url


def _function(tool: JsonDict, gemini: bool) -> JsonDict:
    function = {
        "name": _function_name(tool["name"]),
        "description": tool.get("description") or tool["name"],
    }
    schema = tool.get("input_schema") or {"type": "object"}
    if gemini:
        parameters = gemini_parameters(schema)
        if parameters is not None:
            function["parameters"] = parameters
    else:
        function["parameters"] = schema
    return {"type": "function", "function": function}


def _post(url: str, body: JsonDict, api_key: str | None, timeout: float) -> JsonDict:
    """POST JSON; transient failures raise ``TransientModelError``. Keys never appear."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")[:300]
        if api_key:
            text = text.replace(api_key, "[key]")
        error = TransientModelError if exc.code in {408, 429} or exc.code >= 500 else RuntimeError
        raise error(f"Model request failed with HTTP {exc.code}: {text}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise TransientModelError(f"Model request failed: {reason}") from None


def _arguments(raw: str | None) -> JsonDict:
    try:
        arguments = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"invalid_json": raw}
    return arguments if isinstance(arguments, dict) else {}


def _metered(decision, reply: JsonDict):
    tokens = (reply.get("usage") or {}).get("total_tokens")
    return MeteredDecision(decision, tokens) if isinstance(tokens, int) else decision


def openai_compatible(
    model: str,
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    *,
    timeout: float = 300,
    temperature: float | None = None,
):
    """A model adapter for APIs in the OpenAI Chat Completions format.

    The model sees the tools its node may use as functions and answers with a
    tool call or with text, which becomes the node's ``answer``. Timeouts,
    dropped connections, HTTP 429 and 5xx raise ``TransientModelError``, so the
    framework's automatic retries apply. The key is sent only to ``base_url``
    and never appears in errors.

    Provider state is kept in the step's ``model_state`` and sent back: Gemini's
    thought signatures (Gemini refuses a tool-call history without them; its
    tool schemas are also converted to the subset it accepts) and the
    ``reasoning_content`` of thinking models such as DeepSeek's.

    OpenAI's newest models need the Responses API for tool calls: use
    ``openai_responses`` for them.
    """
    endpoint = base_url.rstrip("/") + "/chat/completions"
    gemini = is_gemini(base_url)

    def adapter(messages: list[JsonDict]) -> Any:
        payload = json.loads(messages[-1]["content"])
        if payload.get("request_type"):
            raise ValueError(f"Unsupported request type {payload['request_type']!r}.")
        names = {_function_name(t["name"]): t["name"] for t in payload.get("available_tools", [])}
        body: JsonDict = {"model": model, "messages": _chat_messages(messages, payload, gemini)}
        if names:
            body["tools"] = [_function(tool, gemini) for tool in payload["available_tools"]]
        if temperature is not None:
            body["temperature"] = temperature
        reply = _post(endpoint, body, api_key, timeout)
        message = reply["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            return _metered(
                NodeResult(values={"answer": (message.get("content") or "").strip()}), reply
            )
        function = calls[0]["function"]
        state: JsonDict = {}
        google = (calls[0].get("extra_content") or {}).get("google") or {}
        if google.get("thought_signature"):
            state["thought_signature"] = google["thought_signature"]
        if message.get("reasoning_content"):
            state["reasoning_content"] = message["reasoning_content"]
        decision = ReActAction(
            f"tool:{names.get(function['name'], function['name'])}",
            _arguments(function.get("arguments")),
            model_state=state or None,
        )
        return _metered(decision, reply)

    return adapter


def _response_items(payload: JsonDict) -> list[JsonDict]:
    """Responses API input: the request, then each step's call and result."""
    items: list[JsonDict] = [{"role": "user", "content": _request_text(payload)}]
    # Reasoning items are bound to the exact earlier input: replay them only while
    # no steps were left out of the prompt and the tools are unchanged.
    replay = not payload.get("history_omitted_older_steps") and not payload.get("react_final_round")
    for number, entry in enumerate(payload.get("history") or []):
        observation = json.dumps(entry.get("observation"), ensure_ascii=False)
        if "action" not in entry:
            items.append(
                {"role": "user", "content": f"Your result was not accepted: {observation}"}
            )
            continue
        state = entry.get("model_state") or {}
        call_id = state.get("call_id") or f"call_{number}"
        if replay and state.get("items"):
            items.extend(state["items"])
        else:
            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": _function_name(entry["action"]["name"].removeprefix("tool:")),
                    "arguments": json.dumps(
                        entry["action"].get("arguments") or {}, ensure_ascii=False
                    ),
                }
            )
        items.append({"type": "function_call_output", "call_id": call_id, "output": observation})
    if payload.get("react_final_round"):
        items.append({"role": "user", "content": FINAL_ROUND_NOTE})
    return items


def openai_responses(
    model: str,
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    *,
    timeout: float = 300,
):
    """A model adapter for OpenAI's Responses API, which OpenAI's newest models
    (GPT-6.2 Astra and later) need for tool calls.

    Requests are stateless (``store: false``); each tool call's reasoning items
    (encrypted) are kept in the step's ``model_state`` and sent back, so the
    model continues its reasoning across tool calls. Errors behave as in
    ``openai_compatible``.
    """
    endpoint = base_url.rstrip("/") + "/responses"

    def adapter(messages: list[JsonDict]) -> Any:
        payload = json.loads(messages[-1]["content"])
        if payload.get("request_type"):
            raise ValueError(f"Unsupported request type {payload['request_type']!r}.")
        tools = payload.get("available_tools") or []
        names = {_function_name(t["name"]): t["name"] for t in tools}
        body: JsonDict = {
            "model": model,
            "instructions": _system_text(messages, payload),
            "input": _response_items(payload),
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "name": _function_name(tool["name"]),
                    "description": tool.get("description") or tool["name"],
                    "parameters": tool.get("input_schema") or {"type": "object"},
                    "strict": False,
                }
                for tool in tools
            ]
            body["parallel_tool_calls"] = False
        reply = _post(endpoint, body, api_key, timeout)
        kept, text = [], []
        for item in reply.get("output") or []:
            kind = item.get("type")
            if kind in ("reasoning", "function_call"):
                kept.append(item)
            if kind == "function_call":
                decision = ReActAction(
                    f"tool:{names.get(item['name'], item['name'])}",
                    _arguments(item.get("arguments")),
                    model_state={"call_id": item.get("call_id"), "items": kept},
                )
                return _metered(decision, reply)
            if kind == "message":
                text += [
                    c.get("text", "")
                    for c in item.get("content") or []
                    if c.get("type") == "output_text"
                ]
        return _metered(NodeResult(values={"answer": "\n".join(text).strip()}), reply)

    return adapter


# -- the terminal chat (python quickstart.py) ------------------------------------


def demo_model():
    """A scripted adapter for checking an installation without an API key: it lists
    the workspace, then answers with what it found."""

    def adapter(messages):
        payload = json.loads(messages[-1]["content"])
        history = payload.get("history") or []
        if not history:
            return ReActAction("tool:fs.list_directory", {"path": "."})
        entries = (history[-1].get("observation") or {}).get("entries") or []
        names = ", ".join(entry.get("name", "?") for entry in entries) or "nothing yet"
        return NodeResult(values={"answer": f"Demo mode works. The workspace contains: {names}."})

    return adapter


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    import sys

    from workflow_model_setup import load_config, setup_model

    parser = argparse.ArgumentParser(
        description="Chat with a Directed Growing Sheet assistant in the terminal."
    )
    parser.add_argument("--demo", action="store_true", help="run once without a model or key")
    parser.add_argument("--setup", action="store_true", help="choose the model (again)")
    parser.add_argument(
        "--workspace",
        default=os.environ.get("DGS_WORKSPACE"),
        help="folder the assistant works in (default: the preset's, usually ./workspace)",
    )
    parser.add_argument("--allow-writes", action="store_true", help="let it write files")
    args = parser.parse_args(argv)

    if args.demo:
        reply = Assistant(demo_model(), args.workspace).ask("What is in the workspace?")
        print(reply.answer if reply.status == "completed" else f"Failed: {reply.error}")
        return 0 if reply.status == "completed" else 1
    if args.setup:
        return 0 if setup_model() else 1

    config = load_config()
    if not config.get("model") or not config.get("base_url"):
        if not sys.stdin.isatty():
            print("No model chosen yet: run python quickstart.py --setup (see GETTING_STARTED.md).")
            return 2
        config = setup_model()
        if not config:
            return 1
    adapter = openai_responses if config.get("api") == "responses" else openai_compatible
    assistant = Assistant(
        adapter(config["model"], config.get("api_key"), config["base_url"]),
        args.workspace,
        allow_writes=args.allow_writes,
    )
    print(f"Model: {config['model']}  (change it with python quickstart.py --setup)")
    print(f"Workspace: {Path(args.workspace).resolve()}")
    print("Type a request, or an empty line to quit.")
    while True:
        try:
            message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            break
        reply = assistant.ask(message)
        while reply.status == "waiting":
            request = reply.pending[0]
            print(f"\nApproval needed: {json.dumps(request['details'], ensure_ascii=False)[:500]}")
            approved = input("Allow? [y/N] ").strip().lower() == "y"
            reply = assistant.decide(reply, request["id"], approved)
        if reply.status == "completed":
            print(f"\nassistant> {reply.answer}")
        else:
            print(f"\nFailed: {reply.error}")
    return 0
