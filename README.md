**English** | [简体中文](README.zh-CN.md)

# Directed Growing Sheet

Directed Growing Sheet (DGS) is a JSON-first workflow engine for
tool-using LLM agents. Workflows are graphs of nodes and edges stored as JSON.
Nodes can be plain Python handlers or ReAct loops driven by a model you
connect. Every action an agent takes goes through one dispatcher that enforces
permissions, risk limits, file and network safety policies and human
approvals, and records the call in a persistent task journal.

> **New to DGS?** Read [GETTING_STARTED.md](GETTING_STARTED.md): install with one
> command and chat with an agent in five minutes. This README is the technical
> reference.

The engine includes no user interface, MCP transport or server. For model
calls it includes one small adapter for OpenAI-compatible APIs
(`openai_compatible`); for anything else you write an adapter.

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Building a workflow by hand](#building-a-workflow-by-hand)
- [Core concepts](#core-concepts)
- [Running tasks](#running-tasks)
- [Nodes and models](#nodes-and-models)
- [Permissions: allowances and risk levels](#permissions-allowances-and-risk-levels)
- [Human checks](#human-checks)
- [Tools](#tools)
- [Built-in tools](#built-in-tools)
- [Declarative tools](#declarative-tools)
- [Edges and routing](#edges-and-routing)
- [Joins and start conditions](#joins-and-start-conditions)
- [Pre-run and post-run hooks](#pre-run-and-post-run-hooks)
- [Restarts, replay and retries](#restarts-replay-and-retries)
- [Timeouts](#timeouts)
- [Memory](#memory)
- [Long-term knowledge](#long-term-knowledge)
- [Agent Skills](#agent-skills)
- [Generated graphs](#generated-graphs)
- [Settings reference](#settings-reference)
- [Security model and limitations](#security-model-and-limitations)
- [Changes from the previous version](#changes-from-the-previous-version)
- [Project layout and development](#project-layout-and-development)

## Requirements

- Python 3.10 or later. The core uses only the standard library.
- `python install.py` creates a virtual environment in `.venv`, installs DGS
  into it with Agent Skills support (PyYAML) and runs a self-check.
- To use DGS from another project, install it with
  `pip install -e path/to/directed-growing-sheet` or put the folder on
  `PYTHONPATH`. The modules read their bundled data (`schemas/`, `tools/`,
  `presets/`, `memory_settings.json`) from that folder, so a non-editable
  install is not supported.

Run the tests and an example:

```bash
python -m unittest discover -s tests
```

```bash
python example_react.py
```

## Quick start

The recommended setup is one command and one class.

```bash
python install.py
```

The installer creates `.venv`, installs DGS, runs a self-check and asks one
question: which model to use. The default is **Gemini 4 Hydrogen**
(`gemini-4-hydrogen`); Claude Opus 5.5 (`claude-opus-5-5`, through
Anthropic's OpenAI-compatible endpoint), GPT-6.2 Astra (`gpt-6.2-astra`,
through the Responses API), DeepSeek V4 Pro (`deepseek-v4-pro`), Ollama, LM
Studio and any other OpenAI-compatible API are offered too. The IDs are the
providers' published ones as of November 2026.
It checks the choice against the models the key can use and with one test
request, and saves it in `.dgs/config.json` (git-ignored, readable only by the
user). `python install.py --skip-model` skips the question, and
`python quickstart.py --setup` asks it again (`workflow_model_setup`).

`python quickstart.py` then opens a chat in the terminal with the saved model
(`DGS_API_KEY`, `DGS_MODEL` and `DGS_BASE_URL` override it; `--workspace`,
`--allow-writes`; `--demo` runs without a model). From Python:

```python
from workflow_core import Assistant, openai_compatible

model = openai_compatible(
    "gemini-4-hydrogen",
    api_key="...",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)
assistant = Assistant(model, workspace="workspace")
reply = assistant.ask("Summarise notes.txt")
print(reply.status, reply.answer)
```

`Assistant` is one LLM node on an engine set up with:

| Default | Meaning |
| --- | --- |
| Built-in toolbox | `BuiltinPrivateToolModule`, confined to the `workspace` folder |
| `safe-default` allowance | `presets/allowances/safe-default.json`: read files in the workspace (`fs.stat`, `fs.list_directory`, `fs.read_text`, `fs.read_json`) and work with text, JSON, CSV, hashes and URLs; risk `low`. Nothing is written, deleted or sent over the network, and no tools can be defined. |
| `allow_writes=True` | Uses `presets/allowances/workspace-editor.json` instead: also `fs.write_text` and `fs.make_directory` (risk `medium`) |
| Automatic retries | `model_retry_attempts = 2` (`retries=`) |
| Step limit | `max_steps = 20`, then one last round to answer (`react.on_step_limit = "answer"`) |

**Every default is JSON** in `presets/`, validated against
`schemas/workflow.schema.json` when it is loaded, so a setup can be changed or
shipped without touching code:

| File | Contents | Schema |
| --- | --- | --- |
| `presets/allowances/*.json` | The allowance sets (`safe-default`, `workspace-editor`); `default_allowances()` loads all of them | `allowance` |
| `presets/assistant.json` | The assistant: its workflow (one LLM node with instructions, step limit and `on_step_limit`), the allowance with and without `allow_writes`, `model_retry_attempts` and the default workspace. `Assistant(..., preset=path)` loads another file; `load_assistant_preset()` reads one | `assistant_preset` |
| `presets/models.json` | The models `install.py` offers: `id`, `label`, `base_url`, `model`, `needs_key`, `key_hint`, `api` (`chat` or `responses`), and the `default` service | `model_services` |

The keyword arguments of `Assistant` (`instructions`, `allow_writes`,
`max_steps`, `retries`, `workspace`) override single values of the preset.

`reply.status` is `completed`, `failed` or `waiting`; for `waiting`, decide
each entry of `reply.pending` with `assistant.decide(reply, request_id,
approved)`. `reply.task` is the DGS task, which can be saved with
`task.save(path)` and continued with `assistant.resume(Task.load(path))`.

For workflows of your own, `create_engine(workspace)` returns an engine with
the same toolbox, presets and retries, and `use_default_allowance(workflow)`
gives every node without an allowance `safe-default` (nodes without an
allowance are otherwise unrestricted). `default_allowances()` returns a
registry with the presets.

**`openai_compatible(model, api_key=None, base_url="https://api.openai.com/v1",
timeout=120, temperature=None)`** sends a node's prompt to
`<base_url>/chat/completions` with the node's tools as functions and turns the
reply into a tool call or the node's `answer`. It uses only the standard
library, works with Gemini (Google's compatible endpoint), DeepSeek, OpenAI and
local servers such as Ollama or LM Studio, raises `TransientModelError` for
timeouts, connection errors, HTTP 408, 429 and 5xx (so retries apply), and never
includes the key in errors. For Gemini it converts tool schemas to the subset
Gemini accepts (`workflow_schema_compat`) and sends each tool call's thought
signature back through the step's `model_state`, as Gemini requires; a thinking
model's `reasoning_content` (DeepSeek) is sent back the same way.

**`openai_responses(model, api_key=None, base_url="https://api.openai.com/v1",
timeout=300)`** does the same through OpenAI's Responses API, which OpenAI's
newest models (GPT-6.2 Astra) need for tool calls. Requests are stateless
(`store: false`); each step's encrypted reasoning items are kept in its
`model_state` and sent back while the prompt has left no steps out. It
handles nodes whose output is `answer`; other output layouts need an adapter of
your own.

## Building a workflow by hand

A one-node workflow whose agent may call a single low-risk tool. The model
adapter is scripted here; in a real application it sends `messages` to your
model and converts the reply into a `ReActAction` or `NodeResult`.

```python
import json

from workflow_core import (
    AllowanceRegistry, AllowanceSet, LocalToolModule, NodeResult, ReActAction,
    Task, ToolDefinition, WorkflowDefinition, WorkflowEngine,
)

workflow = WorkflowDefinition.from_dict({
    "id": "lookup",
    "name": "Answer from the docs",
    "start": ["agent"],
    "edges": [],
    "nodes": [{
        "id": "agent",
        "type": "agent",
        "inputs": {"query": "/query"},
        "outputs": {"answer": "/answer"},
        "allowance": "reader",
    }],
})

tools = LocalToolModule()
tools.register(
    ToolDefinition("docs.search", risk_level="low"),
    lambda context, tool, args: {"text": f"Refunds take 5 days ({args['q']})."},
)
allowances = AllowanceRegistry()
allowances.register(AllowanceSet("reader", actions=["tool:docs.search"], max_tool_risk="low"))
engine = WorkflowEngine(allowances, tools=tools)


def model(messages):
    """Send `messages` to your model and turn its reply into a decision (scripted here)."""
    request = json.loads(messages[-1]["content"])
    if request.get("request_type") == "active_memory":
        return "Searched the docs for the user's question."
    if not request["history"]:
        return ReActAction("tool:docs.search", {"q": request["inputs"]["query"]})
    return NodeResult(values={"answer": request["history"][-1]["observation"]["text"]})


engine.register_llm("agent", model)
task = Task.create(workflow, {"query": "refund time"})
engine.task_pool.add(task)
while ready := engine.task_pool.ready():
    for ready_task, execution in ready:
        engine.run_execution(workflow, ready_task, execution)

print(task.status, task.data["answer"])
task.save("task.json")  # git-ignored in this folder
```

## Core concepts

| Object | Role |
| --- | --- |
| `WorkflowDefinition` | Portable workflow: `nodes`, `edges`, `start` node IDs. Loaded from and saved to JSON. |
| `NodeDefinition` | One node: `type` (selects the handler), `inputs`/`outputs` (JSON Pointers into task data), and the optional settings described below. |
| `EdgeDefinition` | A directed edge with an optional `id`, `when` condition, `otherwise` flag, `entry_check` and `max_traversals`. |
| `Task` | The persistent runtime object: `data`, `executions`, `status`, the event journal (`events`) and `active_memory`. There is no separate state object. |
| `TaskExecution` | One run of one node. Parallel branches are several executions in the same task. |
| `NodeContext` | The runtime view passed to handlers, hooks and tools: inputs, the task, the execution and methods such as `use_tool`. |
| `NodeResult` | What a handler returns: `status` (`completed`, `waiting` or `failed`), `values` (named outputs), `local` and `error`. |
| `WorkflowEngine` | Runs executions, dispatches actions, schedules successors. |
| `TaskPool` | All tasks the engine knows about; needed for joins across tasks. |

All definitions are validated against `schemas/workflow.schema.json` (JSON Schema
Draft 2020-12) when loaded. Unknown fields are rejected, so a typo such as
`otherwize` fails loudly. Free-form objects (`config`, `ui`, hook configs, task
data) accept any JSON. When building definitions in Python, call
`workflow.assert_valid()`; the engine also validates before each run.

A node writes outputs only to the paths declared in `outputs`. A result that
names an undeclared output fails the execution.

## Running tasks

`engine.run_execution(workflow, task, execution)` runs one execution whose
status is `ready`:

1. pre-run hook, then the handler (or the ReAct loop);
2. outputs are applied to `task.data`;
3. post-run hook, then the optional post-run human review;
4. outgoing edges are evaluated and successor executions created.

Each step can pause (`waiting`) or fail. Handler, hook, input and scheduling
errors fail the execution with a message in `execution.error`; they do not
raise. `run_execution` raises only when it is misused: the execution is not
`ready`, or no handler is registered for the node type.

Execution statuses: `ready`, `running`, `waiting` (a human decision or other
application input), `waiting_start` (an arrival held by a join), `completed`,
`failed`, `skipped` and `joined`. Task statuses: `running`, `completed`,
`failed` and `merged` (consumed into a cross-task join). A task becomes
`completed` or `failed` once no execution can make progress.

`engine.task_pool.ready()` returns every `(task, execution)` pair that can run,
including tasks created by joins. Tasks are added to the pool when they first
run, or explicitly with `engine.task_pool.add(task)`.

**Following progress.** `engine.add_run_observer(callback)` calls
`callback(task, execution)` before every node execution runs, including the
executions of generated child graphs, which run inside the parent's
`graph.create` call on their own task. Hosts use it to show live progress: keep
the task objects it reports and read their executions and journal while the
run continues. Observers must not change the task; one that raises is recorded
as an `observer.failed` event and the run goes on.

**Persistence** is up to the host. `task.save(path)` writes `path` and a sidecar
folder `<file name>.memory/` holding the journal and active memory as
checksum-verified JSONL. Copy both together. Old stream files are deleted when a
new snapshot no longer uses them. `Task.load(path)` restores the task.
`TaskPool.save/load` does the same for a whole pool; use it when tasks join
across task boundaries. Register handlers and hooks again after loading: Python
functions are never stored in JSON.

## Nodes and models

### Plain handlers

```python
engine.register("form.check", lambda context: NodeResult(values={"ok": True}))
```

A handler receives a `NodeContext` and returns a `NodeResult`. Use
`context.use_tool(name, arguments)` or `context.act(name, arguments)` for any
operation that should be permission-checked and journaled.

### ReAct loops

```python
engine.register_react("agent", reason, observe=None)
```

`reason(context, history)` returns a `ReActAction(name, arguments)` to run an
action, or a `NodeResult` to finish, wait or fail. Each decision uses one of the
node's `react.max_steps` rounds (default 10 per invocation); running out fails
the node. Actions and observations persist in `execution.local["react"]`;
private model reasoning is never stored.

An adapter may attach opaque `model_state` (a JSON object) to a `ReActAction`:
data the provider needs to continue after the action, such as an encrypted
reasoning item, thinking blocks or a thought signature. It is stored with the
step, returned in later prompts' history exactly as given (never shortened or
budgeted), and never written to the journal.

A failed action (denied permission, unknown tool, tool exception) is added to
the history as `{"observation": {"error": {"type": ..., "message": ...}}}` so the
model can try something else. A final result naming undeclared outputs is
returned to the model the same way. Set `"react": {"on_action_error": "fail"}`
on a node to fail it on the first such error instead. Pauses for human approval
and node timeouts always stop the loop.

`react.max_steps` (default 10) counts model decisions, the final result
included, and starts again after each pause for approval. When it runs out the
node fails. With `"react": {"on_step_limit": "answer"}` the model is asked once
more instead, with no tools and `react_final_round: true` in the prompt, so it
can return a result from what it already found; the node fails only if it still
asks for an action. The journal records `react.final_round`.

A node's summary is not proof that it did its work. `react.result_check` names a
host check that verifies the real state before the result counts:

```python
def all_screened(context, result):
    left = library.count(status="candidate")
    return None if left == 0 else f"{left} candidates are still undecided; decide them all."

engine.register_result_check("all_screened", all_screened)
# node: "react": {"max_steps": 30, "result_check": "all_screened"}
```

The check returns None to accept the result, or an explanation. A rejected
result is not applied: the explanation goes back to the model as an
observation (`ResultRejected`) and the loop continues, each rejection using a
step. The final round of `on_step_limit: "answer"` is checked too; a node that
runs out of steps without an accepted result fails with the explanation, so
the next nodes never start on unfinished work. The journal records
`react.result_rejected`. Generated graphs may name a check only if it was
registered with `generated_graphs=True`.

**Long nodes stay small.** Every step is kept in full in the execution's
journal, but a prompt carries only the newest `react_full_steps` steps
(default 5) and every failed step (an error, or a result a check rejected) in
full. Older steps become a numbered index, one line each; each complete block
of `react_index_fold` steps (default 10; 0 never folds) becomes one line:

```text
#1-#10: called library.next_candidates ×5, library.decide ×5
#11 library.decide - decided 25: 3 included, 22 excluded
#12 library.next_candidates failed (in full below)
```

The model reads any archived step again with the built-in `history.read` tool
(`{"steps": "11"}`, `"3,9"` or `"1-10"`, up to 10 per call). It is offered once
steps are archived and needs no allowance entry: it only shows the node its
own steps. A tool can describe its calls for the index:
`module.register(tool, handler, summary=lambda args, result: "...")`; others
get a line from their arguments and result. Index lines never change once
written, so the start of the prompt stays stable for provider prompt caches.
Set `react.full_steps` and `react.index_fold` on a node to change both for that
node.

### Connecting a model

```python
engine.register_llm("agent", decide, memory_writer=None, branch_selector=None)
```

The adapter receives a list of chat messages (`role`, `content`) and is called
for three kinds of request, distinguished by `request_type` in the JSON user
message:

| Request | Return value |
| --- | --- |
| node decision (no `request_type`) | `ReActAction` or `NodeResult`, optionally wrapped in `MeteredDecision(decision, total_tokens)` |
| `"active_memory"` | plain text, or `{"text": ..., "references": [event IDs]}` |
| `"branch_selection"` | `{"edges": [edge IDs], "reason": "..."}` |

A fourth request, `"knowledge_consolidation"`, goes to a separate callback; see
[Long-term knowledge](#long-term-knowledge).

Pass `memory_writer` or `branch_selector` to use separate callbacks for the last
two, or `memory_writer=False` to write no active-memory notes for that node type
(useful for short chat turns; objective checkpoints are still recorded). The framework makes no assumption about the model vendor or SDK.

The decision prompt contains the node definition, the resolved allowance, the
tools the node may see (`available_tools`, with input schemas, risk levels and
human-check flags), the hook list, the start condition, the inputs, pre-run
data, join provenance, a bounded view of the task memory and the ReAct history.
A custom `register` or `register_react` handler can build the same messages with
`context.prompt_messages(history)` and send them itself.

**Automatic retries.** When a model call times out, loses its connection or
finds the provider overloaded, the adapter should raise
`TransientModelError` (or an exception with `transient = True`;
`TimeoutError` and `ConnectionError` count too). With
`"model_retry_attempts": N` in the settings, the framework waits and calls the
adapter again up to N times, `model_retry_delay_seconds` at first and twice as
long each time, at most `model_retry_max_delay_seconds`, and stops early if
the node's time runs out. Each retry is journaled as `llm.retry`
(`knowledge.retry`, `memory.retry` for the writers) with the error. Other
errors are never retried. `0`, the default, switches retries off. A host can
switch them per engine without touching the file:
`engine.setting_overrides = {"model_retry_attempts": 3}`.

## Permissions: allowances and risk levels

An allowance is a reusable JSON file referenced by ID from nodes:

```json
{"id": "reader", "actions": ["tool:fs.read_text", "tool:mcp:docs:search"], "max_tool_risk": "low", "config": {}}
```

```python
allowances = AllowanceRegistry()
allowances.load("examples/allowances/read_only.json")
engine = WorkflowEngine(allowances)
```

- A node without `allowance` (or with `null`) may use every registered action
  and tool. An allowance with `"actions": []` allows nothing.
- Action names match exactly. Tools are named `tool:<name>`; MCP tools are
  `tool:mcp:<server>:<name>`.
- Every tool has an application-assigned `risk_level`: `low` < `medium` <
  `high` < `critical`. A tool is allowed only if its action is listed **and** its
  risk is at or below `max_tool_risk` (no limit when omitted).
- `allowance_check(context, allowance, action, arguments)`, passed to
  `WorkflowEngine`, can enforce argument-level rules from the allowance's
  `config`. It must return `True` to allow.
- The registry is read on every call: re-register or reload a file to change
  permissions immediately. A missing allowance ID fails the node.

Permissions govern the dispatcher (`context.act`/`use_tool` and model actions).
Handlers, hooks and tool backends are trusted Python and are not sandboxed.

## Human checks

No human checks are enabled by default. Available gates:

- **Tool approval**: set `"human_check": true` on a tool definition (or
  `tools.set_human_check(name)`). The call pauses after the permission checks and
  before execution.
- **Post-run review**: set `"post_run_human_check": true` on a node. After its
  outputs and post-run hook, the execution pauses before choosing outgoing edges.
- **Custom checkpoints**: call `require_human_check(context, key, details)` in
  a hook before a protected operation.
- File deletion, non-GET HTTP requests, generated graphs, skill scripts and
  timeouts add their own approvals (see the sections below).

When a gate is reached, the execution becomes `waiting` and the task stays
active. The host shows the request and records the decision:

```python
from workflow_core import pending_human_checks, resolve_human_check

for request in pending_human_checks(task):
    ...  # show request["details"] to an authenticated person
resolve_human_check(task, request_id, approved=True, reviewer="user-42", comment="Checked")
engine.run_execution(workflow, task, execution)  # the execution is ready again
```

An approval is single-use and bound to the exact details it showed (tool
definition, arguments, allowance reference, file metadata and so on); if they
change, a new request is created. Approval never overrides permissions: they are
checked again before execution. Rejecting fails the execution, and it cannot be
retried, unless the request's details contain `"on_reject": "continue"`: then
the execution becomes ready again and the protected operation sees the
rejection (`require_human_check` raises `PermissionError`) and can skip itself.
Saving a graph into a skill uses this. Keep `resolve_human_check` away from agent tools; the framework does not
provide identity, notifications or a user interface (`example_human_checks.py`
uses terminal input).

## Tools

### Local tools

```python
tools = LocalToolModule()
tools.register(ToolDefinition("documents.lookup", risk_level="low"), handler)
engine.attach_tools(tools)   # or WorkflowEngine(tools=tools); attach_tools(None) detaches
```

`handler(context, tool, arguments)` returns JSON-compatible data. Arguments are
checked against the tool's `input_schema` before the handler runs; an invalid
call fails with a message naming the bad field (`arguments.limit: must be <= 5`). Tool metadata
(`name`, `risk_level`, `description`, `input_schema`, `human_check`, `config`)
round-trips through `ToolDefinition.save/load`.

Any object with `get_tool(name)` and `execute(context, tool, arguments)` can be
the backend (`ToolModule` protocol); add `list_tools()` so models can see the
tools. `get_tool` must not have side effects. The abstract base classes
`ToolBackendInterface`, `PrivateToolModule` and `MCPToolModule` describe the
same contract; an optional `authorize_call(context, tool, arguments)` hook runs
after the permission checks and must keep the human check
(`require_tool_human_check`).

### Several backends

The engine has one local tool backend. To combine several, for example the
built-in tools and a module holding skills, wrap them:

```python
engine.attach_tools(CompositeToolModule(BuiltinPrivateToolModule(workspace), skills_module))
```

Tool names must be unique across members; each call goes to the member that
owns the tool, including its safety hook.

### MCP servers

MCP adapters are attached by a stable ID; the host owns connection, credentials
and shutdown. The framework provides only the `MCPToolModule` interface, not an
MCP client.

```python
engine.attach_mcp_server("github", github_adapter)
```

A node lists the servers it uses in `"requires": {"mcp_servers": ["github"]}`;
only those servers' tools are visible and callable from that node, as
`tool:mcp:github:<tool>`.

## Built-in tools

`BuiltinPrivateToolModule(workspace_root)` provides 20 tools described in
`tools/private_tools.json` (descriptions, risk levels and human-check flags can
be edited; argument names and types must match the implementations):

| Group | Tools | Risk |
| --- | --- | --- |
| Files | `fs.stat`, `fs.list_directory`, `fs.read_text`, `fs.read_json` | low |
| Files | `fs.write_text` (no overwrite unless `overwrite: true`), `fs.make_directory` | medium |
| Files | `fs.delete_file` (one regular file) | high |
| Text | `text.search`, `text.replace`, `text.split`, `text.join` | low |
| JSON | `json.parse`, `json.stringify`, `json.get` | low |
| CSV | `csv.parse`, `csv.format` | low |
| Other | `hash.sha256`, `url.build` | low |
| Other | `http.request`, `tools.define` | high (always) |

```python
tools = BuiltinPrivateToolModule(
    "C:/project/workspace",
    http_allowed_origins=["https://api.example.com"],
    http_credentials={"https://api.example.com": {"Authorization": "Bearer ..."}},
    on_security_event=show_alert,           # optional UI notification
)
```

**Paths** must be relative to the workspace and stay inside it after resolving
links; absolute paths, `..` escapes, Windows drive and stream syntax are
rejected. This is a path check, not an operating-system sandbox.

**Limits**: reads and writes default to 262144 bytes, directory listings to 200
entries (`max_read_bytes`, `max_write_bytes`, `max_entries`). `tools.configure({...})`
replaces the whole configuration; omitted keys return to their defaults.

**Safety locks** (`tools/safety.json`, or `safety_config_path=`):

```json
{"delete_requires_human": true, "block_system_edits": true, "additional_protected_paths": []}
```

- `delete_requires_human`: `fs.delete_file` always needs approval. The approval is
  bound to the resolved path and the file's metadata; if the file changes while
  waiting, a new approval is needed.
- `block_system_edits`: writes, directory creation and deletion under system
  directories (Windows, Program Files, ProgramData, boot files; `/etc`, `/usr`,
  `/bin` and similar) and under `additional_protected_paths` are refused. Approval
  cannot override this.
- Security events are stored in `execution.local["tool_security_events"]` and the
  journal, and sent to `on_security_event` (a Python warning without one). The
  callback's return value is ignored; it never grants permission.

**HTTP**: only HTTPS origins listed in `http_allowed_origins` (empty by default)
are reachable. DNS results must all be public addresses; the connection is
pinned to the checked address and TLS is verified against the host name.
Redirects are not followed. Transport headers (`Host`, `Content-Length` and
similar) are rejected. `POST`, `PUT`, `PATCH` and `DELETE` need human approval.
The socket timeout is 15 seconds.

**Responses** are prepared for a model:

| Field | Meaning |
| --- | --- |
| `status`, `headers`, `content_type` | As sent by the server |
| `encoding` | Charset used to decode the body: the declared one (`Content-Type`, or an HTML `<meta charset>`), else UTF-8 |
| `body` | The decoded body; with `"format": "text"`, an HTML page's visible text (no markup, scripts or styles) |
| `title` | The HTML title, with `"format": "text"` |
| `truncated` | The body was cut at `max_read_bytes` (previously the call failed) |
| `note` | Present when the response looks like a block or bot challenge (status 401/403/429/503, CAPTCHA or "Just a moment..." pages) |

Requests carry a `User-Agent` and an `Accept` header unless the caller or the
host's credentials set them. `Accept-Encoding` is managed by the runtime (the
model cannot ask for compression); a gzip or deflate body sent anyway is
decompressed, never past `max_read_bytes`.

**Credentials** belong in `http_credentials`, never in tool arguments: they are
added to requests for their origin, cannot be overridden by the model and never
appear in the journal or prompts. Anything a model passes as an argument is
recorded and shown to models later.

## Declarative tools

A model with `tool:tools.define` can create reusable tools named `decl.<name>`
from JSON recipes over the built-in tools; no code is generated. See
`examples/declarative_http_tool.json`.

- Up to 20 sequential steps. Arguments may reference inputs with
  `{"$input": "query"}` and earlier results with `{"$step": "fetch", "path": "/body"}`;
  references keep their JSON type. All declared parameters are required.
- The HTTP URL and method are fixed in the definition; query, headers and body
  may use references.
- Saving runs strict schema validation and static analysis (reference order, JSON
  Pointers, known result fields, inferable types, literal paths, HTTP targets).
  Every step tool must be allowed by the current node's allowance.
  `declarative_allowed_tools` in `configure` narrows the tools a recipe may use.
- Running `decl.x` requires `tool:decl.x` and, for every step, that step's own
  permission; steps go through the same checks and approvals as direct calls.
  `decl.x` itself is `high` risk.
- Definitions are stored in `<workspace>/.dgs-tools/definitions.json`, which the
  file tools cannot touch. Editing requires `overwrite: true`.
- If a step waits for approval, finished steps are kept and the run continues
  from the next step. If a step fails, the run is blocked until the host inspects
  the effects and removes the entry from `execution.memory_state["declarative_runs"]`.

## Edges and routing

**Conditions** (`when`):

```json
{"and": [{"eq": ["/type", "invoice"]}, {"gt": ["/amount", 1000]}, {"exists": "/customer/email"}]}
```

Operators: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `exists`, `and`, `or`, `not`.
A string starting with `/` is a JSON Pointer into task data; wrap a literal in
`{"literal": "/usr/bin"}`. Reading a missing path is an error that fails the
source execution; guard optional fields with `exists`.

**Fallbacks**: edges with `"otherwise": true` are used only when no ordinary
edge passes.

**Entry checks**: a named Python predicate, kept out of the JSON:

```python
engine.register_entry_check("files_created", lambda context, config: ...)  # must return bool
```

An edge needs both its `when` condition and its entry check to pass. Unknown
names, exceptions and non-Boolean returns fail the source execution without
creating any successor.

**Traversal limits**: `"max_traversals": n` caps how often an edge may be
entered in a task (`0` disables it). The count comes from the journal, so it
survives save and load. Every edge needs a unique ID; give parallel edges an
explicit `id`.

**Routing** (`node.routing.mode`), applied among the edges that passed:

- `default`: take `default_edges` if given (an empty list takes none); otherwise
  every passing edge.
- `llm`: with more than one candidate, the branch selector returns edge IDs and a
  reason; the reason is stored in active memory.
- `human`: with more than one candidate, the execution waits. Use
  `pending_branch_selections(task)` and
  `resolve_branch_selection(task, request_id, edge_ids, reviewer=...)`.

Edges that are not taken produce `skipped` arrivals downstream, so joins can
tell that a branch will never arrive. Skipped nodes run no handler, hook or tool.

## Joins and start conditions

A node without a start condition runs once per arrival. A start condition holds
arrivals until one has arrived on each required edge:

```json
{"id": "merge", "type": "agent.merge",
 "start_condition": {"scope": "group", "group_by": "/batch", "required_edges": ["left-to-merge", "right-to-merge"]}}
```

- `required_edges` must list every incoming edge (default: all of them).
  `"join": "all"` is shorthand for a task-scoped join. A join node cannot be a
  start node.
- `scope: "task"` joins branches of one task. `scope: "group"` joins different
  tasks whose data has the same non-null value at `group_by`; their data is
  merged into a new task (disjoint keys combine, equal values are kept, any
  conflict fails the join with its path). Source tasks become `merged`.
- `generation_by` separates rounds of a loop; without it, arrivals pair up first
  in, first out per edge.
- `check` names a predicate registered with `engine.register_start_check`; it
  can keep the join waiting.
- Skipped arrivals count as resolved for task joins and, with
  `"accept_skipped": true`, for group joins. If every arrival was skipped, the
  join node is skipped too.

Call `engine.refresh_start_conditions(workflow)` after changing something a start
check depends on. The pool is a single-threaded coordinator: hosts with several
workers must serialise scheduling and persist the pool atomically.

## Pre-run and post-run hooks

```json
{"id": "agent", "type": "agent.summarize",
 "pre_run": "inspect_folder", "pre_run_config": {"pattern": "*.txt"},
 "post_run": "write_summary", "post_run_config": {"filename": "summary.txt"}}
```

```python
engine.register_pre_run("inspect_folder", lambda context, config: {"files": [...]})
engine.register_post_run("write_summary", lambda context, result, config: None)
```

- `pre_run(context, config)` runs before the handler on every attempt and returns
  a JSON object (or `None`) available as `context.pre_run_data` and shown to the
  model.
- `post_run(context, result, config)` runs after outputs are applied and before
  edges are evaluated, only when the node completed. It must return `None`.
- Both hook names are resolved before anything runs, so a missing post-run hook
  cannot be silently skipped.
- Hook errors fail the execution; a failed post-run hook can be retried without
  rerunning the handler (see below).

## Restarts, replay and retries

A Python call stack cannot be saved. When a plain handler or hook pauses for a
human decision, it runs again from the top once the decision arrives. To keep
that safe, every top-level action that completed is written to a journal for
the current phase (`execution.local["action_journal"]`). When the function runs
again, calls with the same name and arguments in the same order are answered
from the journal: they do not run twice and do not ask for approval again. The
first call that differs ends replay and runs normally. ReAct nodes do not need
this; their history already records each action, and a paused action resumes
exactly as reviewed.

`engine.retry_execution(task, execution, replay=True)` makes a failed execution
ready again:

- a failure in the post-run hook or while scheduling resumes from that step; the
  handler does not run again;
- otherwise the handler reruns, with completed actions replayed (pass
  `replay=False` to run everything again);
- executions with a rejected approval cannot be retried.

Replay is not exactly-once delivery: a crash between a tool's side effect and the
next save can still repeat it. Make external effects idempotent where it matters.

## Timeouts

```json
{"id": "review", "type": "agent", "timeout_seconds": 120, "timeout_human_check": true}
```

The deadline starts when the execution is created and includes time spent
waiting for joins and approvals. It is checked cooperatively at tool calls, model
calls and lifecycle steps; call `engine.check_timeouts(workflow)` periodically
from your scheduler, especially while nodes wait for people. When it passes:

- with `timeout_human_check: true` (default), a `timeout` approval is created;
  approving grants another `timeout_seconds` from that moment;
- otherwise the execution fails.

A blocking Python, HTTP or model call cannot be interrupted; give clients their
own timeouts, or use process isolation if you need a hard stop.

## Memory

| Layer | Lives in | Written by | Lifetime |
| --- | --- | --- | --- |
| Objective memory | `task.events` | framework and host | one task |
| Active memory | `task.active_memory` | the model, citing events | one task |
| Knowledge | `KnowledgeStore` | reviewed proposals | across tasks ([next section](#long-term-knowledge)) |
| Procedures | skills, declarative tools, saved graphs | host and reviewed model requests | across tasks |

**Objective memory** is the task journal `task.events`, written only by the
framework and the host. Each event has `id`, `timestamp`, `sequence`, `task_id`,
`execution_id`, `node_id`, `type` and `details`. It records task and node
lifecycle, every action and tool call (arguments, results, errors), hooks, edge
decisions, joins, approvals, security events, model calls (not hidden reasoning)
and skill activity. There is no switch to turn it off. Hosts add their own
events with `record_event(task, "application.note", {...})` or
`context.record_activity(kind, details)`; browsers can use `ObservedBrowser` or
`record_page_opened`. After a cross-task join, `sequence` is local to each
source task, so do not sort merged journals by it alone.

**Active memory** (`task.active_memory`) holds notes written by the model at
checkpoints. Every note cites objective event IDs; when they disagree, the
journal wins. Checkpoints are created at node start, after each tool call, at
the round and token intervals from the settings, at node summaries and at
host-declared names (`memory_checkpoints` on the node, then
`context.checkpoint("name", details)`). Without a memory writer the checkpoint
stays queued; no note is invented. A node summary moves the notes it replaces to
`active_memory["archive"]`; `read_memory_archive(task, ...)` reads them.

**What models see.** Prompts never contain the whole journal. They get a bounded
view: the newest events (bookkeeping such as `llm.started` left out) and the
newest current notes, each section within a character budget, with long strings
cut short. The view reports how much was left out. The ReAct history is bounded
the same way, and the memory writer receives only the evidence for its
checkpoint. Limits are in [the settings file](#settings-reference); because every
section has a budget, prompts stop growing however long the task runs.

## Long-term knowledge

Objective and active memory belong to one task. A `KnowledgeStore` keeps what
should outlive tasks: facts and lessons, stored as **pages** of short
**entries** in a folder the host owns. It uses no embeddings and no vector
database. Run `python example_knowledge.py` for a complete example.

```python
store = KnowledgeStore("C:/project/knowledge")        # keep it outside the agent workspace
engine = WorkflowEngine(tools=tools, knowledge=store, knowledge_writer=writer)
```

```json
{"id": "agent", "type": "agent", "knowledge": {"read": ["global", "support"], "propose": ["support"]}}
```

**Scopes** group pages (`global`, `support`, one per workflow, one per customer;
lower-case names). A node without a `knowledge` setting sees and reaches
nothing. `propose` scopes must also be listed in `read`.

**Reading.** The node's prompt contains a generated index (page name, title,
one-line summary) within a character budget, and the node gets three low-risk
tools: `knowledge.list`, `knowledge.read` and `knowledge.search` (keyword
ranking; Chinese, Japanese and Korean text is matched by character pairs).
Pages are named `<scope>/<slug>`, for example `support/refunds`. The tools go
through the node's allowance like any other tool, and reads are journaled.

**Writing is always indirect.** Nothing edits a page in place. A change is a
list of small operations:

| Operation | Fields |
| --- | --- |
| `create_page` | `page`, `title`, `summary` |
| `set_summary` | `page`, `summary`, optional `title` |
| `add_entry` | `page`, `text` (one self-contained fact or lesson) |
| `update_entry` | `page`, `entry` (ID), `text` |
| `retire_entry` | `page`, `entry` (ID), `reason` |

A change is validated and dry-run when proposed; all of its operations apply
together or not at all. Applying increases the page version and writes the
before and after documents to `log.jsonl`. Changes come from three places:

- **Consolidation** (the main path). When `run_execution` settles a task as
  completed or failed, the `knowledge_writer` callback receives a request with
  `request_type: "knowledge_consolidation"` (a bounded view of the task, the
  index, related pages and the proposals still waiting for review, so it does
  not propose them twice) and returns
  `{"operations": [...], "rationale": "...", "evidence": [event IDs]}`, or an
  empty `operations` list. A failing writer is journaled and never fails the
  task. Call `engine.consolidate_knowledge(workflow, task)` yourself for tasks
  that end another way.
- **`knowledge.propose`** (medium risk), for nodes with `propose` scopes; limited
  per task.
- **The host**: `store.apply(operations, author="admin")` applies immediately.

**Review.** Proposals wait in the store and never pause the task:

```python
for proposal in store.proposals():                    # pending, oldest first
    ...                                               # show operations, rationale and source
store.resolve_proposal(proposal["id"], True, reviewer="ann", comment="Matches policy")
store.revert(change_id, reviewer="ann")               # undo one applied change
```

Every proposal records its task, the journal events it cites and whether the
task used outside content. A task is **external** if it ran any action other
than the pure data tools (`text.*`, `json.*`, `csv.*`, `hash.sha256`,
`url.build`), the knowledge tools, or actions the host vouches for in
`engine.knowledge_trusted_actions` (for example `"tool:docs.search"`).
Declarative tools and generated graphs are judged by the steps they run. Proposals
from external tasks always wait for a person. Setting `knowledge_human_review`
to `false` applies only the others automatically.

**Maintenance.** `store.lint()` reports duplicate entries, oversized, empty and
unused pages and stale proposals; it changes nothing. `store.changes()` lists
the change log. Several processes may share one store folder.

**Limits.** The index-in-prompt approach suits up to a few hundred pages per
node; past that, split scopes so each node reads fewer. Search is keyword
matching, not semantic similarity: write entries with the words people will
search for. Generated graphs inherit the creating node's read scopes and can
never propose.

## Agent Skills

```python
skills = LocalToolModule()
register_skill(skills, "./skills/review", scripts=["scripts/check.py"])
engine.attach_tools(CompositeToolModule(BuiltinPrivateToolModule(workspace), skills))
```

A skill package follows the [Agent Skills specification](https://agentskills.io/specification):
a folder whose `SKILL.md` has YAML frontmatter with a `name` matching the folder.
Registering adds these tools:

| Tool | Risk | Purpose |
| --- | --- | --- |
| `skill.<name>.invoke` | low | Open an invocation; returns the instructions and the graphs saved in the package |
| `skill.<name>.read` | low | Read a file of the package (UTF-8 or base64) |
| `skill.<name>.finish` | low | Close the invocation |
| `skill.<name>.script` | high, human check | Run a registered `scripts/*.py` (only if scripts were registered) |
| `skill.<name>.run_graph` | high | Run a graph saved in the package as a reviewed child task (see below) |

Each tool goes through the node's allowance like any other. Invocations can be
nested with `parent_invoke_id`; a parent closes after its children. Skill
activity is journaled. Scripts run as `python -I -` with the verified bytes on
stdin, in the package folder, with a 30-second default timeout; the first 16 KiB
of output is kept. They are trusted code, not sandboxed: they can reach files
and the network directly. `allowed-tools` in the manifest is metadata only.

## Generated graphs

A node with `"allow_create_graph": true` and permission for `tool:graph.create`
(risk `high`) may ask to run a small workflow of its own: `nodes`, `edges`,
`start` and an `explanation` (see `examples/generated_graph_request.json`).
After review it runs immediately as a child task with a copy of the parent's
data; the call returns the child's data.

Limits:

- Every child node's allowance must be a subset of the creating node's
  allowance (actions and risk ceiling) and of every enclosing graph's; omitted
  allowances inherit the parent's. The parent's argument check also applies.
  MCP servers must be a subset of the parent's.
- **Host code needs an opt-in.** Node types, pre/post-run hooks, entry checks
  and start checks are Python. A graph may only use names registered with
  `generated_graphs=True`, for example
  `engine.register_post_run("notify", hook, generated_graphs=True)`. LLM node
  types (`register_llm`) are allowed by default because they act only through
  checked actions; plain `register` and `register_react` types are not.
- Child nodes inherit the creating node's knowledge read scopes and can never
  propose knowledge changes.
- Human branch routing is not supported inside generated graphs.
- Depth, approvals and the step budget come from the settings file.
- Approvals requested inside the child (tools, timeouts) are forwarded to the
  parent's reviewer.

Rejecting a graph in review does not fail the creating node: the
`graph.create` call fails with the reviewer's comment, so the model can carry
on without the graph or propose another (the review is marked
`"on_reject": "continue"`; the journal records `graph.rejected`). Proposing the
same graph again is refused the same way.

### Graphs saved in skills

A skill can keep the graphs it was run with, so the next run can start from a
graph that worked:

1. **Save.** With an open skill invocation, `"save_to_skill": true` and a
   Markdown `usage_scenario` (when to use the graph) save it to the skill's
   `graphs/<id>.json` and `graphs/<id>.md`. The save is reviewed separately,
   and only after the graph completed successfully; a failed graph is never
   offered. Declining the save keeps the graph's result (`saved_path` is then
   `null`).
2. **Find.** `skill.<name>.invoke` returns `graphs`: one entry per saved graph,
   newest first, with `id`, `title` (the scenario's first line), `saved_at`,
   the `usage_scenario` (first 1500 characters), `explanation` and node ids
   and names. A file whose scenario no longer matches its recorded SHA-256, or
   whose graph is invalid, is listed with `invalid` and the reason instead.
3. **Reuse.** `skill.<name>.run_graph` with the open `invoke_id` and a
   `graph_id` runs that graph through the package's `SkillGraphInterface`.
   The default implementation loads and verifies the file and runs it as a
   generated graph of the calling node: child allowances must be subsets of
   the node's, it is reviewed when `graph_creation_human_check` is on (the
   review shows `skill_graph`), and depth and step limits apply. The node does
   not need `allow_create_graph`: the `run_graph` action in its allowance is
   the permission, and only graphs already in the package can run.

To change how saved graphs run (for example on another worker), subclass
`SkillGraphInterface`, override `execute_graph(context, invoke_id, graph_id)`
and pass an instance to `register_skill(..., graph_runner=runner)`. Pass
`graph_runner=None` to leave `run_graph` out.

Any file in `graphs/` with the same format works, so a skill can also ship
graphs written by hand. `load_skill_graph(root, id)` and
`list_skill_graphs(root)` do the verification for hosts.

#### Environment and run history

A graph that worked before may not work in a different setup. The framework
keeps `graphs/<id>.meta.json` next to each saved graph and updates it itself,
without the model: when the graph is saved, and after every run through
`run_graph` (each run is counted once, also across a review pause).

| Field | Contents |
| --- | --- |
| `saved` | Time of saving and the environment then |
| `runs` | `total`, `completed`, `failed` |
| `last_run` | Time, `outcome`, `error`, node runs (`steps`), `child_task_id`, environment |
| `last_success` | Time and environment of the latest completed run |
| `graph_sha256` | The graph file it describes; metadata for another version is ignored |

The **environment** is what the graph depends on: whether its node types are
registered, digests of the allowances its nodes use and of the tools those
allowances permit, the host's `engine.host_environment` (facts such as
`{"model": "anthropic/claude-opus-5-5"}`, set by the host), and the Python
version.

When the skill is opened, every listed graph also gets:

- `history`: `saved_at`, `runs` and the last run's time, outcome and error;
- `environment`: `status` and `changes` from comparing the current environment
  with the last successful run (or the saving): `same`; `changed` (a tool or
  allowance definition or a host fact differs); `incompatible` (a node type,
  allowance or tool is gone); or `unknown` (no run recorded);
- `hint`: one sentence for the model, for example "The environment changed
  since the last successful run (model: provider/a → provider/b). Reuse it
  only if these changes do not matter for the task, and check its result."

The metadata is advisory and not covered by the graph's hash. Deleting it only
loses the history.

## Settings reference

`memory_settings.json` (or `WorkflowEngine(memory_settings_path=...)`) is read
again on every use. Omitted keys take the defaults; unknown keys are errors.

| Key | Default | Meaning |
| --- | --- | --- |
| `react_rounds` | 5 | Checkpoint every N model decisions of one execution |
| `token_limit` | 8192 | Checkpoint each time this many decision tokens are used (provider counts via `MeteredDecision`, otherwise estimated). This is a checkpoint interval, not a spending cap. |
| `prompt_max_events` | 100 | Most recent journal events shown to a model |
| `prompt_max_value_chars` | 1000 | Longer strings in prompt memory and history are shortened |
| `prompt_budget_chars` | 30000 | Character budget for each prompt section: events, notes, history (a Chinese, Japanese or Korean character counts four) |
| `react_full_steps` | 5 | Newest ReAct steps sent in full; older ones become an index (failed steps always in full) |
| `react_index_fold` | 10 | Fold each complete block of this many archived steps into one index line (0: never) |
| `prompt_knowledge_chars` | 6000 | Character budget for the knowledge index in a prompt |
| `knowledge_human_review` | true | Every knowledge proposal waits for a person; `false` auto-applies proposals from tasks without outside content |
| `knowledge_consolidate_on_finish` | true | Ask the knowledge writer for lessons when a task completes or fails |
| `knowledge_max_proposals_per_task` | 5 | `knowledge.propose` calls allowed per task |
| `graph_max_depth` | 3 | Maximum nesting of generated graphs (0 disables them) |
| `graph_creation_human_check` | true | Review each generated graph before it runs |
| `graph_save_human_check` | true | Review saving a graph into a skill |
| `graph_max_steps` | 100 | Node runs allowed per generated graph |
| `model_retry_attempts` | 0 | Retries of a model call that failed transiently (0 = off) |
| `model_retry_delay_seconds` | 2 | Wait before the first retry; doubled for each further one |
| `model_retry_max_delay_seconds` | 30 | Longest wait between retries |

## Security model and limitations

- **Trusted code**: handlers, hooks, entry and start checks, tool backends and
  skill scripts are trusted Python. Permissions control what goes through the
  dispatcher, not what that code does directly.
- **Not a sandbox**: path checks do not defend against hard links or files
  replaced concurrently; use operating-system isolation for untrusted work.
- **Prompt injection**: tool output, memory and knowledge are presented to models
  as data, with instructions saying so, but a model can still be misled.
  Permissions, approvals and safety locks are the real boundary. Long-term
  knowledge makes an injection durable if it is accepted, which is why
  proposals from tasks that used outside content always need a person.
- **Concurrency**: the engine and pool are synchronous and single-threaded.
  Hosts running several workers must serialise scheduling and use atomic
  persistence; there are no distributed locks.
- **Durability**: snapshots are written atomically, but events since the last
  save are lost if the process crashes.
- **Not included**: model clients, MCP transport, user interface, identity,
  notifications. `ToolCreationInterface`, `SkillWriteInterface` and
  `SkillAgentInterface` are interfaces only.

## Changes from the previous version

- The project is now called Directed Growing Sheet (DGS). Module names
  (`workflow_*`) are unchanged; the package is `directed-growing-sheet`.
- New recommended setup: `python install.py` (asks only which model to use,
  Gemini 4 Hydrogen by default), with all defaults as JSON in `presets/`
  (schemas `assistant_preset`, `model_services`), `Assistant`, the
  `openai_compatible` adapter, `create_engine`, `use_default_allowance` and the
  `safe-default` / `workspace-editor` allowance presets.

- Errors in handlers, inputs, results and scheduling fail the execution instead
  of raising or leaving it stuck in `running`. `run_execution` no longer raises
  for scheduling errors. Use `engine.retry_execution` instead of setting statuses
  by hand.
- ReAct nodes show failed actions to the model instead of failing; set
  `"react": {"on_action_error": "fail"}` for the old behaviour.
- Generated graphs may use only node types, hooks and checks registered with
  `generated_graphs=True` (LLM node types are allowed by default).
- Prompts contain a bounded view of memory, controlled by the new settings
  `prompt_max_events`, `prompt_max_value_chars` and `prompt_budget_chars`.
  `node.run_finished` events no longer copy the task data.
- Restarted handlers and hooks replay completed calls instead of repeating them.
- Rejecting an approval or failing on a timeout now settles the task status.
- `CompositeToolModule` combines tool backends; skill package registrations are
  in `module.skill_packages` (previously `_skill_packages`).
- New: long-term knowledge (`KnowledgeStore`, the `knowledge.*` tools and
  consolidation). It is off unless a store is attached and a node opts in.
- HTTP credentials are configured with `http_credentials`.
- Conditions accept `{"literal": value}`; the template folder check ignores case.
- `http.request` decodes bodies with the declared charset (not always UTF-8),
  truncates long bodies instead of failing, decompresses gzip/deflate, sends a
  `User-Agent`, flags block pages, and has a `"format": "text"` mode for HTML.
- New: `engine.add_run_observer` reports every node execution, including those
  of generated child graphs, for progress displays.
- Automatic model-call retries (`model_retry_*` settings, off by default,
  `TransientModelError`), and per-engine `engine.setting_overrides`.
- `react.on_step_limit: "answer"` gives a node one last, tool-less round when
  `max_steps` runs out, instead of failing it.
- Rejecting a generated graph in review no longer fails the creating node; the
  model gets the reviewer's comment and carries on.
- New: long ReAct nodes send only their newest steps (`react_full_steps`, 5)
  and failed steps in full; older steps become an index (folded per
  `react_index_fold`, 10) that the model reads back with `history.read`. Tools
  can summarize their calls for the index (`register(..., summary=...)`). The
  prompt budget counts CJK characters as four.
- `graph.create` shows the model a compact schema (the node fields a generated
  graph can use); proposals are still validated against the full schema.
- New: `react.result_check` and `engine.register_result_check` let the host
  verify a ReAct node's work before its result is applied; rejected results go
  back to the model with the reason.
- Graphs saved in skills are listed by `skill.<name>.invoke` and run with the
  new `skill.<name>.run_graph` tool; `SkillGraphInterface` is implemented and
  replaceable. Each saved graph keeps its environment and run history in
  `graphs/<id>.meta.json`, and the listing says when the environment changed
  (`engine.host_environment` declares host facts such as the model). Saving now happens after the graph completed, and declining it
  no longer fails the node (`"on_reject": "continue"`).

## Project layout and development

| Path | Contents |
| --- | --- |
| `workflow_core.py` | Public API; import from here |
| `workflow_quickstart.py` | `Assistant`, `openai_compatible`, `openai_responses`, `create_engine` |
| `presets/` | The defaults of the recommended setup, as JSON: `allowances/`, `assistant.json`, `models.json` |
| `install.py`, `quickstart.py`, `workflow_model_setup.py` | One-step setup with the model choice, and the terminal chat |
| `workflow_schema_compat.py` | Tool schemas in the subset Gemini accepts |
| `GETTING_STARTED.md` | Short guide for new users |
| `workflow_engine.py` | Execution, dispatch and scheduling |
| `workflow_definitions.py`, `workflow_executions.py`, `workflow_nodes.py` | Data model |
| `workflow_react.py`, `workflow_prompts.py`, `workflow_routing.py` | Agent loop, prompts, branch selection |
| `workflow_allowances.py`, `workflow_approvals.py`, `workflow_timeouts.py`, `workflow_replay.py` | Permissions, approvals, deadlines, replay |
| `workflow_tools.py`, `workflow_private_tools.py`, `workflow_basic_tools.py`, `workflow_http_tools.py`, `workflow_declarative_*.py`, `workflow_tool_safety.py` | Tools |
| `workflow_logging.py`, `workflow_memory.py`, `workflow_memory_store.py` | Journal, active memory, snapshots |
| `workflow_knowledge.py`, `workflow_knowledge_tools.py` | Long-term knowledge store, tools and consolidation |
| `workflow_joins.py`, `workflow_start_conditions.py` | Joins |
| `workflow_skills.py`, `workflow_graph_tools.py`, `workflow_graphs.py`, `workflow_skill_graph_meta.py` | Skills, generated graphs, saved graphs' environment and history |
| `schemas/workflow.schema.json` | JSON Schema for all definitions |
| `tools/` | Built-in tool catalog and safety settings |
| `examples/`, `example_*.py` | Runnable examples |
| `tests/` | Unit tests |

Formatting and linting use [ruff](https://docs.astral.sh/ruff/) with the settings
in `pyproject.toml`:

```bash
ruff format . && ruff check .
```

## License

Directed Growing Sheet is licensed under the [Apache License 2.0](LICENSE).
