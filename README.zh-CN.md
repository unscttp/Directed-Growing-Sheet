[English](README.md) | **简体中文**

# Directed Growing Sheet

Directed Growing Sheet（DGS）是面向调用工具的 LLM 智能体的工作流引擎，以 JSON 为核心格式。工作流是由节点和边组成的图，
以 JSON 保存。节点可以是普通 Python 处理函数，也可以是由你接入的模型驱动的 ReAct 循环。
智能体的每个动作都经过同一个调度入口：它负责检查权限、风险上限、文件与网络安全策略和人工审批，
并把调用写入可持久化的任务日志。

> **第一次使用 DGS？** 请阅读 [GETTING_STARTED.zh-CN.md](GETTING_STARTED.zh-CN.md)：一条命令完成安装，五分钟内和智能体对话。
> 本 README 是技术参考文档。

引擎不包含用户界面、MCP 传输层或服务端。模型调用方面，它自带一个面向 OpenAI 兼容 API 的小型适配器（`openai_compatible`）；
其他情况需要你自己编写适配器。

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [手动构建工作流](#手动构建工作流)
- [核心概念](#核心概念)
- [运行任务](#运行任务)
- [节点与模型](#节点与模型)
- [权限 allowance 与风险等级](#权限-allowance-与风险等级)
- [人工审批](#人工审批)
- [工具](#工具)
- [内置工具](#内置工具)
- [声明式工具](#声明式工具)
- [边与路由](#边与路由)
- [汇合与启动条件](#汇合与启动条件)
- [前置与后置 hook](#前置与后置-hook)
- [重启重放与重试](#重启重放与重试)
- [超时](#超时)
- [记忆](#记忆)
- [长期知识](#长期知识)
- [Agent Skills](#agent-skills)
- [模型生成的子图](#模型生成的子图)
- [设置项](#设置项)
- [安全模型与限制](#安全模型与限制)
- [与上一版本的差异](#与上一版本的差异)
- [项目结构与开发](#项目结构与开发)

## 环境要求

- Python 3.10 或更高版本。核心只使用标准库。
- `python install.py` 会在 `.venv` 中创建虚拟环境，把 DGS 连同 Agent Skills 支持（PyYAML）安装进去，并运行自检。
- 在其他项目中使用时，执行 `pip install -e path/to/directed-growing-sheet`，或把该目录加入 `PYTHONPATH`。
  各模块从该目录读取自带的数据（`schemas/`、`tools/`、`presets/`、`memory_settings.json`），因此不支持非 editable 安装。

运行测试和示例：

```bash
python -m unittest discover -s tests
```

```bash
python example_react.py
```

## 快速开始

推荐的用法只需要一条命令和一个类。

```bash
python install.py
```

安装程序会创建 `.venv`、安装 DGS、运行自检，然后只问一个问题：使用哪个模型。默认是 **Gemini 4 Hydrogen**（`gemini-4-hydrogen`）；
也可以选择 Claude Opus 5.5（`claude-opus-5-5`，通过 Anthropic 的 OpenAI 兼容接口）、GPT-6.2 Astra（`gpt-6.2-astra`，通过 Responses API）、
DeepSeek V4 Pro（`deepseek-v4-pro`）、Ollama、LM Studio 或其他任何 OpenAI 兼容的 API。这些 ID 是各服务商截至 2026 年 11 月公布的模型 ID。它会对照该密钥可用的模型检查所选模型，
并发送一次测试请求，然后保存到 `.dgs/config.json`（git 会忽略它，且只有当前用户可读）。
`python install.py --skip-model` 跳过这个问题，`python quickstart.py --setup` 可以重新选择（`workflow_model_setup`）。

之后 `python quickstart.py` 会用保存的模型在终端中打开对话（`DGS_API_KEY`、`DGS_MODEL` 和 `DGS_BASE_URL` 可以覆盖保存的设置；
另有 `--workspace`、`--allow-writes`；`--demo` 不需要模型即可运行）。在 Python 中使用：

```python
from workflow_core import Assistant, openai_compatible

model = openai_compatible(
    "gemini-4-hydrogen",
    api_key="...",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)
assistant = Assistant(model, workspace="workspace")
reply = assistant.ask("总结 notes.txt")
print(reply.status, reply.answer)
```

`Assistant` 是一个 LLM 节点，运行在按以下默认设置配置好的引擎上：

| 默认设置 | 含义 |
| --- | --- |
| 内置工具箱 | `BuiltinPrivateToolModule`，限定在 `workspace` 目录内 |
| `safe-default` allowance | `presets/allowances/safe-default.json`：读取工作目录中的文件（`fs.stat`、`fs.list_directory`、`fs.read_text`、`fs.read_json`），处理文本、JSON、CSV、哈希和 URL；风险等级 `low`。不会写入、删除文件或访问网络，也不能定义新工具。 |
| `allow_writes=True` | 改用 `presets/allowances/workspace-editor.json`：另外允许 `fs.write_text` 和 `fs.make_directory`（风险等级 `medium`） |
| 自动重试 | `model_retry_attempts = 2`（参数 `retries=`） |
| 步数上限 | `max_steps = 20`，用完后再给一个回合作答（`react.on_step_limit = "answer"`） |

**所有默认设置都是 JSON**，保存在 `presets/` 中，加载时按 `schemas/workflow.schema.json` 校验，因此修改或分发一套配置都不需要改代码：

| 文件 | 内容 | Schema |
| --- | --- | --- |
| `presets/allowances/*.json` | allowance 集合（`safe-default`、`workspace-editor`）；`default_allowances()` 会加载其中全部文件 | `allowance` |
| `presets/assistant.json` | 助手：它的工作流（一个 LLM 节点，含说明、步数上限和 `on_step_limit`）、启用与不启用 `allow_writes` 时的 allowance、`model_retry_attempts` 以及默认工作目录。`Assistant(..., preset=path)` 加载其他文件；`load_assistant_preset()` 读取预设 | `assistant_preset` |
| `presets/models.json` | `install.py` 提供的模型：`id`、`label`、`base_url`、`model`、`needs_key`、`key_hint`、`api`（`chat` 或 `responses`），以及默认服务 `default` | `model_services` |

`Assistant` 的关键字参数（`instructions`、`allow_writes`、`max_steps`、`retries`、`workspace`）可以覆盖预设中的单个值。

`reply.status` 为 `completed`、`failed` 或 `waiting`；为 `waiting` 时，用 `assistant.decide(reply, request_id, approved)`
逐个处理 `reply.pending` 中的请求。`reply.task` 是 DGS 任务，可以用 `task.save(path)` 保存，
再用 `assistant.resume(Task.load(path))` 继续。

构建自己的工作流时，`create_engine(workspace)` 返回带有同样工具箱、预设和重试设置的引擎；
`use_default_allowance(workflow)` 会给所有没有 allowance 的节点设置 `safe-default`（否则这些节点不受限制）。
`default_allowances()` 返回包含这些预设的注册表。

**`openai_compatible(model, api_key=None, base_url="https://api.openai.com/v1", timeout=120, temperature=None)`**
把节点的提示词发送到 `<base_url>/chat/completions`，把节点可用的工具作为函数提供，再把回复转换成工具调用或节点的 `answer`。
它只使用标准库，适用于 Gemini（Google 的兼容接口）、DeepSeek、OpenAI 以及 Ollama、LM Studio 等本地服务。
对 Gemini，它会把工具 schema 转换为 Gemini 接受的子集（`workflow_schema_compat`），并通过步骤的 `model_state`
把每次工具调用的 thought signature 发回（这是 Gemini 的要求）；思考模型的 `reasoning_content`（DeepSeek）也会以同样方式发回。

**`openai_responses(model, api_key=None, base_url="https://api.openai.com/v1", timeout=300)`** 通过 OpenAI 的 Responses API
完成同样的工作；OpenAI 最新的模型（GPT-6.2 Astra）进行工具调用时需要它。请求不在服务端保存（`store: false`）；
每一步加密的推理条目保存在该步骤的 `model_state` 中，只要提示词没有省略步骤，就会原样发回。遇到超时、连接错误以及 HTTP 408、429 和 5xx 时
抛出 `TransientModelError`（因此会自动重试），错误信息中不会包含密钥。它处理输出名为 `answer` 的节点；其他输出结构需要自行编写适配器。

## 手动构建工作流

下面是只有一个节点的工作流，其中的智能体只能调用一个低风险工具。示例中的模型适配器是写死的脚本；
实际应用里，它应把 `messages` 发给你的模型，再把回复转换成 `ReActAction` 或 `NodeResult`。

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
    """把 messages 发给模型，并把回复转换成决策（这里用脚本代替）。"""
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
task.save("task.json")  # 本目录中已被 git 忽略
```

## 核心概念

| 对象 | 作用 |
| --- | --- |
| `WorkflowDefinition` | 可移植的工作流：`nodes`、`edges` 和起始节点 `start`，可与 JSON 互相转换。 |
| `NodeDefinition` | 单个节点：`type` 决定处理函数，`inputs`/`outputs` 是指向任务数据的 JSON Pointer，其余为下文介绍的可选设置。 |
| `EdgeDefinition` | 有向边，可选 `id`、`when` 条件、`otherwise` 标记、`entry_check` 和 `max_traversals`。 |
| `Task` | 持久化的运行对象：`data`、`executions`、`status`、事件日志 `events` 和 `active_memory`。没有单独的状态对象。 |
| `TaskExecution` | 某个节点的一次运行。并行分支就是同一任务中的多个 execution。 |
| `NodeContext` | 传给处理函数、hook 和工具的运行时视图：输入、任务、execution，以及 `use_tool` 等方法。 |
| `NodeResult` | 处理函数的返回值：`status`（`completed`、`waiting` 或 `failed`）、`values`（具名输出）、`local` 和 `error`。 |
| `WorkflowEngine` | 运行 execution、调度动作、安排后继节点。 |
| `TaskPool` | 引擎已知的全部任务；跨任务汇合时需要它。 |

所有定义在加载时都会按 `schemas/workflow.schema.json`（JSON Schema Draft 2020-12）校验。
未知字段会被拒绝，因此 `otherwize` 这样的拼写错误会直接报错。自由格式的对象
（`config`、`ui`、hook 配置、任务数据）可以包含任意 JSON。在 Python 中直接构造定义时，
请调用 `workflow.assert_valid()`；引擎每次运行前也会校验。

节点只能写入 `outputs` 中声明的路径。结果中出现未声明的输出名时，该 execution 失败。

## 运行任务

`engine.run_execution(workflow, task, execution)` 运行一个状态为 `ready` 的 execution：

1. 前置 hook，然后是处理函数（或 ReAct 循环）；
2. 把输出写入 `task.data`；
3. 后置 hook，然后是可选的后置人工审核；
4. 评估出边并创建后继 execution。

每一步都可能暂停（`waiting`）或失败。处理函数、hook、输入和调度中的错误会让 execution 失败，
错误信息写在 `execution.error`，不会抛出异常。只有调用方式不当时 `run_execution` 才会抛出异常：
execution 不是 `ready`，或节点类型没有注册处理函数。

execution 的状态有：`ready`、`running`、`waiting`（等待人工决定或其他应用输入）、
`waiting_start`（被汇合条件暂存的到达）、`completed`、`failed`、`skipped` 和 `joined`。
任务的状态有：`running`、`completed`、`failed` 和 `merged`（已并入跨任务汇合）。
当没有任何 execution 还能推进时，任务变为 `completed` 或 `failed`。

`engine.task_pool.ready()` 返回所有可运行的 `(task, execution)`，包括汇合产生的新任务。
任务在第一次运行时自动加入任务池，也可以用 `engine.task_pool.add(task)` 显式加入。

**跟踪进度。** `engine.add_run_observer(callback)` 会在每个节点 execution 运行前调用 `callback(task, execution)`，
包括生成图的子任务中的 execution（子图在父节点的 `graph.create` 调用内部、以独立的任务运行）。宿主可以用它显示实时进度：
保留回调报告的任务对象，在运行过程中读取它们的 execution 和日志。观察者不能修改任务；抛出异常的观察者会被记录为
`observer.failed` 事件，运行照常继续。

**持久化**由宿主决定。`task.save(path)` 写入 `path`，以及旁边的 `<文件名>.memory/` 目录，
其中以带校验和的 JSONL 保存日志和主动记忆。复制时两者要一起复制。新快照不再引用的旧数据文件会被删除。
`Task.load(path)` 用于恢复任务。`TaskPool.save/load` 对整个任务池做同样的事；任务需要跨任务汇合时使用它。
加载后要重新注册处理函数和 hook：Python 函数不会写入 JSON。

## 节点与模型

### 普通处理函数

```python
engine.register("form.check", lambda context: NodeResult(values={"ok": True}))
```

处理函数接收 `NodeContext`，返回 `NodeResult`。凡是需要权限检查和日志记录的操作，
都应通过 `context.use_tool(name, arguments)` 或 `context.act(name, arguments)` 进行。

### ReAct 循环

```python
engine.register_react("agent", reason, observe=None)
```

`reason(context, history)` 返回 `ReActAction(name, arguments)` 以执行动作，或返回 `NodeResult`
以结束、等待或失败。每次决策消耗一轮 `react.max_steps`（每次调用默认 10 轮），轮数用尽则节点失败。
动作和观察结果保存在 `execution.local["react"]`；模型的内部推理不会被保存。

适配器可以在 `ReActAction` 上附加不透明的 `model_state`（JSON 对象）：模型服务在该动作之后继续对话所需的数据，
例如加密的推理条目、thinking 块或 thought signature。它随该步骤保存，在之后提示词的历史中原样返回
（不会被截短，也不计入预算），并且永远不会写入日志。

动作失败时（权限被拒、未知工具、工具抛出异常），错误会以
`{"observation": {"error": {"type": ..., "message": ...}}}` 的形式加入历史，让模型换一种做法。
最终结果中出现未声明的输出名时，也会以同样方式反馈给模型。在节点上设置
`"react": {"on_action_error": "fail"}` 可改为遇到第一个此类错误就让节点失败。
等待人工审批和节点超时总会中止循环。

`react.max_steps`（默认 10）统计模型的决策次数（包括最终结果），每次因等待审批而暂停后重新计数。用完时节点失败。
设置 `"react": {"on_step_limit": "answer"}` 后，会改为再询问模型一次：不提供任何工具，并在提示中标明 `react_final_round: true`，
让模型根据已经得到的信息给出结果；只有它仍然要求执行动作时节点才会失败。日志中会记录 `react.final_round`。

节点给出的总结并不能证明它确实完成了工作。`react.result_check` 指定一个由宿主提供的检查，在结果生效之前核对实际状态：

```python
def all_screened(context, result):
    left = library.count(status="candidate")
    return None if left == 0 else f"{left} candidates are still undecided; decide them all."

engine.register_result_check("all_screened", all_screened)
# 节点："react": {"max_steps": 30, "result_check": "all_screened"}
```

检查返回 None 表示接受结果，否则返回说明。被拒绝的结果不会生效：说明会作为观察结果（`ResultRejected`）返回给模型，
循环继续，每次拒绝占用一步。`on_step_limit: "answer"` 的最后一个回合同样会被检查；步数用完仍没有被接受的结果时，
节点会带着这条说明失败，后续节点就不会在未完成的工作上开始。日志中会记录 `react.result_rejected`。
生成图只能使用以 `generated_graphs=True` 注册的检查。

**长节点的提示词保持精简。** 每一步都会完整保存在 execution 的日志中，但提示词只完整携带最新的 `react_full_steps` 步（默认 5）
以及所有失败的步骤（出错的调用，或被检查拒绝的结果）。更早的步骤变成编号索引，每步一行；每满 `react_index_fold` 步
（默认 10；0 表示不折叠）的完整区块折叠为一行：

```text
#1-#10: called library.next_candidates ×5, library.decide ×5
#11 library.decide - decided 25: 3 included, 22 excluded
#12 library.next_candidates failed (in full below)
```

模型可以用内置的 `history.read` 工具重新读取任何已归档的步骤（`{"steps": "11"}`、`"3,9"` 或 `"1-10"`，每次最多 10 步）。
有步骤被归档后才会提供这个工具，它不需要 allowance：它只向节点展示节点自己的步骤。工具可以为索引描述自己的调用：
`module.register(tool, handler, summary=lambda args, result: "...")`；其他工具会根据参数和结果生成一行。
索引行一旦写入就不再改变，因此提示词的开头在各次请求之间保持稳定，服务商的提示缓存可以继续生效。
在节点上设置 `react.full_steps` 和 `react.index_fold` 可以单独调整该节点的这两个值。

### 接入模型

```python
engine.register_llm("agent", decide, memory_writer=None, branch_selector=None)
```

适配器接收聊天消息列表（`role`、`content`），会被用于三类请求，
通过用户消息 JSON 中的 `request_type` 区分：

| 请求 | 返回值 |
| --- | --- |
| 节点决策（没有 `request_type`） | `ReActAction` 或 `NodeResult`，可包在 `MeteredDecision(decision, total_tokens)` 中 |
| `"active_memory"` | 纯文本，或 `{"text": ..., "references": [事件 ID]}` |
| `"branch_selection"` | `{"edges": [边 ID], "reason": "..."}` |

第四类请求 `"knowledge_consolidation"` 由单独的回调处理，见[长期知识](#长期知识)。

可以通过 `memory_writer` 或 `branch_selector` 为后两类请求指定单独的回调；传入
`memory_writer=False` 则该节点类型不写主动记忆笔记（适合简短的对话轮次；客观检查点仍会记录）。
框架不依赖任何特定的模型厂商或 SDK。

决策提示词包含：节点定义、解析后的 allowance、节点可见的工具（`available_tools`，含参数格式、
风险等级和人工审批标记）、hook 列表、启动条件、输入、前置数据、汇合来源、受限大小的任务记忆视图
以及 ReAct 历史。自定义的 `register` 或 `register_react` 处理函数可以用
`context.prompt_messages(history)` 生成同样的消息，再自行发送给模型。

**自动重试。** 模型调用超时、连接中断或服务商过载时，适配器应抛出 `TransientModelError`（或带有 `transient = True` 的异常；
`TimeoutError` 和 `ConnectionError` 也算）。设置中 `"model_retry_attempts": N` 时，框架会等待后再次调用适配器，最多 N 次：
第一次等待 `model_retry_delay_seconds`，之后每次加倍，最长 `model_retry_max_delay_seconds`；节点时间用完时提前停止。
每次重试都会以 `llm.retry`（知识整理和记忆写入分别为 `knowledge.retry`、`memory.retry`）连同错误一起记入日志。
其他错误永远不会重试。默认值 `0` 表示关闭重试。宿主可以不改文件、按引擎开关：
`engine.setting_overrides = {"model_retry_attempts": 3}`。

## 权限 allowance 与风险等级

allowance 是可复用的 JSON 文件，节点通过 ID 引用：

```json
{"id": "reader", "actions": ["tool:fs.read_text", "tool:mcp:docs:search"], "max_tool_risk": "low", "config": {}}
```

```python
allowances = AllowanceRegistry()
allowances.load("examples/allowances/read_only.json")
engine = WorkflowEngine(allowances)
```

- 节点没有设置 `allowance`（或设为 `null`）时，可以使用所有已注册的动作和工具。
  `"actions": []` 表示什么都不允许。
- 动作名精确匹配。工具的动作名是 `tool:<名称>`；MCP 工具是 `tool:mcp:<server>:<名称>`。
- 每个工具都有由应用指定的 `risk_level`：`low` < `medium` < `high` < `critical`。
  只有动作名在列表中**并且**风险不高于 `max_tool_risk`（省略时不设上限）时，工具才被允许。
- 传给 `WorkflowEngine` 的 `allowance_check(context, allowance, action, arguments)`
  可以根据 allowance 的 `config` 执行参数级规则，返回 `True` 才放行。
- 每次调用都会重新读取注册表：重新注册或重新加载文件即可立即修改权限。
  引用了不存在的 allowance ID 时，节点失败。

权限约束的是调度入口（`context.act`/`use_tool` 和模型发起的动作）。处理函数、hook 和工具后端
属于受信任的 Python 代码，不在沙箱中运行。

## 人工审批

默认不启用任何人工审批。可用的审批点：

- **工具审批**：在工具定义上设置 `"human_check": true`（或调用 `tools.set_human_check(name)`）。
  调用在权限检查之后、执行之前暂停。
- **后置审核**：在节点上设置 `"post_run_human_check": true`。输出和后置 hook 完成后，
  在选择出边之前暂停。
- **自定义检查点**：在 hook 中受保护的操作之前调用 `require_human_check(context, key, details)`。
- 删除文件、非 GET 的 HTTP 请求、模型生成的子图、skill 脚本和超时会各自产生审批（见下文）。

到达审批点时，execution 变为 `waiting`，任务保持活动状态。宿主展示请求并记录决定：

```python
from workflow_core import pending_human_checks, resolve_human_check

for request in pending_human_checks(task):
    ...  # 把 request["details"] 展示给已认证的用户
resolve_human_check(task, request_id, approved=True, reviewer="user-42", comment="Checked")
engine.run_execution(workflow, task, execution)  # execution 已重新变为 ready
```

每次批准只能使用一次，并绑定到审批时展示的具体内容（工具定义、参数、allowance 引用、文件元数据等）；
内容变化时会生成新的请求。批准不会绕过权限：执行前会再次检查。拒绝会让 execution 失败，且不能重试；
但如果请求的 details 中包含 `"on_reject": "continue"`，execution 会重新变为 ready，受保护的操作会看到拒绝
（`require_human_check` 抛出 `PermissionError`），并可以跳过自身。把图保存到 skill 时就使用这种方式。
不要把 `resolve_human_check` 暴露给智能体工具；框架不提供身份认证、通知或用户界面
（`example_human_checks.py` 使用终端输入演示）。

## 工具

### 本地工具

```python
tools = LocalToolModule()
tools.register(ToolDefinition("documents.lookup", risk_level="low"), handler)
engine.attach_tools(tools)   # 或 WorkflowEngine(tools=tools)；attach_tools(None) 用于卸载
```

`handler(context, tool, arguments)` 返回可序列化为 JSON 的数据。参数会在处理函数运行前按工具的
`input_schema` 校验；不合法的调用会失败，并指出出错的字段（如 `arguments.limit: must be <= 5`）。工具元数据
（`name`、`risk_level`、`description`、`input_schema`、`human_check`、`config`）
可通过 `ToolDefinition.save/load` 保存和读取。

任何实现了 `get_tool(name)` 和 `execute(context, tool, arguments)` 的对象都可以作为后端
（`ToolModule` 协议）；再实现 `list_tools()`，模型才能看到这些工具。`get_tool` 不能有副作用。
抽象基类 `ToolBackendInterface`、`PrivateToolModule` 和 `MCPToolModule` 描述的是同一约定；
可选的 `authorize_call(context, tool, arguments)` 在权限检查之后运行，并且必须保留人工审批
（`require_tool_human_check`）。

### 多个后端

引擎只有一个本地工具后端。需要组合多个后端时（例如内置工具加上存放 skill 的模块），用它包装：

```python
engine.attach_tools(CompositeToolModule(BuiltinPrivateToolModule(workspace), skills_module))
```

各成员的工具名不能重复；每次调用都会交给拥有该工具的成员处理，包括它的安全检查。

### MCP server

MCP 适配器通过稳定的 ID 挂载；连接、凭据和关闭由宿主负责。框架只提供 `MCPToolModule` 接口，
不包含 MCP 客户端。

```python
engine.attach_mcp_server("github", github_adapter)
```

节点在 `"requires": {"mcp_servers": ["github"]}` 中列出要使用的 server；
只有这些 server 的工具对该节点可见、可调用，动作名为 `tool:mcp:github:<工具>`。

## 内置工具

`BuiltinPrivateToolModule(workspace_root)` 提供 20 个工具，说明在 `tools/private_tools.json` 中
（说明、风险等级和人工审批标记可以修改；参数名和类型必须与实现一致）：

| 类别 | 工具 | 风险 |
| --- | --- | --- |
| 文件 | `fs.stat`、`fs.list_directory`、`fs.read_text`、`fs.read_json` | low |
| 文件 | `fs.write_text`（除非 `overwrite: true`，否则不覆盖）、`fs.make_directory` | medium |
| 文件 | `fs.delete_file`（单个普通文件） | high |
| 文本 | `text.search`、`text.replace`、`text.split`、`text.join` | low |
| JSON | `json.parse`、`json.stringify`、`json.get` | low |
| CSV | `csv.parse`、`csv.format` | low |
| 其他 | `hash.sha256`、`url.build` | low |
| 其他 | `http.request`、`tools.define` | high（固定） |

```python
tools = BuiltinPrivateToolModule(
    "C:/project/workspace",
    http_allowed_origins=["https://api.example.com"],
    http_credentials={"https://api.example.com": {"Authorization": "Bearer ..."}},
    on_security_event=show_alert,           # 可选的界面通知
)
```

**路径**必须相对于工作目录，并且在解析链接后仍位于工作目录内；绝对路径、`..` 越界、
Windows 盘符和数据流语法都会被拒绝。这是路径检查，不是操作系统沙箱。

**限制**：读写默认上限各为 262144 字节，目录列表最多 200 项
（`max_read_bytes`、`max_write_bytes`、`max_entries`）。`tools.configure({...})`
会整体替换配置，省略的项恢复默认值。

**安全锁**（`tools/safety.json`，或通过 `safety_config_path=` 指定）：

```json
{"delete_requires_human": true, "block_system_edits": true, "additional_protected_paths": []}
```

- `delete_requires_human`：`fs.delete_file` 总是需要审批。审批绑定解析后的路径和文件元数据；
  等待期间文件发生变化，需要重新审批。
- `block_system_edits`：拒绝在系统目录（Windows、Program Files、ProgramData、启动文件；
  `/etc`、`/usr`、`/bin` 等）以及 `additional_protected_paths` 下写入、创建目录和删除。
  人工批准也无法绕过。
- 安全事件保存在 `execution.local["tool_security_events"]` 和日志中，并发送给
  `on_security_event`（未设置时发出 Python warning）。回调的返回值会被忽略，不会被当作授权。

**HTTP**：只能访问 `http_allowed_origins` 中列出的 HTTPS origin（默认为空）。DNS 解析结果必须全部是
公网地址；连接固定到检查过的地址，并按主机名校验 TLS。不跟随重定向。拒绝 `Host`、`Content-Length`
等传输层请求头。`POST`、`PUT`、`PATCH` 和 `DELETE` 需要人工审批。socket 超时为 15 秒。

**响应**会整理成适合模型阅读的形式：

| 字段 | 含义 |
| --- | --- |
| `status`、`headers`、`content_type` | 服务器返回的内容 |
| `encoding` | 解码正文所用的字符集：服务器声明的字符集（`Content-Type` 或 HTML 的 `<meta charset>`），否则为 UTF-8 |
| `body` | 解码后的正文；指定 `"format": "text"` 时为 HTML 页面的可见文字（去掉标记、脚本和样式） |
| `title` | 指定 `"format": "text"` 时为 HTML 标题 |
| `truncated` | 正文在 `max_read_bytes` 处被截断（以前会直接报错） |
| `note` | 响应看起来是拦截页或人机验证时出现（状态码 401/403/429/503、验证码或“Just a moment...”页面） |

请求会带上 `User-Agent` 和 `Accept` 请求头，除非调用方或宿主的凭据已经设置。`Accept-Encoding` 由运行时管理（模型不能要求压缩）；
服务器仍返回 gzip 或 deflate 压缩的正文时会被解压，解压结果不会超过 `max_read_bytes`。

**凭据**应放在 `http_credentials` 中，不要放在工具参数里：凭据会自动加到对应 origin 的请求上，
模型无法覆盖，也不会出现在日志或提示词中。模型作为参数传入的任何内容都会被记录，并在之后展示给模型。

## 声明式工具

拥有 `tool:tools.define` 权限的模型可以用基于内置工具的 JSON 配方创建可复用工具 `decl.<名称>`；
不会生成任何代码。示例见 `examples/declarative_http_tool.json`。

- 最多 20 个顺序步骤。参数可以用 `{"$input": "query"}` 引用输入，用
  `{"$step": "fetch", "path": "/body"}` 引用之前步骤的结果；引用保留原始 JSON 类型。
  声明的参数全部必填。
- HTTP 的 URL 和 method 在定义中固定；query、请求头和请求体可以使用引用。
- 保存时会进行严格的 schema 校验和静态分析（引用顺序、JSON Pointer、已知结果字段、可推导的类型、
  字面量路径、HTTP 目标）。每个步骤用到的工具都必须被当前节点的 allowance 允许。
  `configure` 中的 `declarative_allowed_tools` 可以进一步缩小配方可用的工具范围。
- 运行 `decl.x` 需要 `tool:decl.x` 权限，并且每个步骤都需要该步骤自身的权限；
  各步骤经过与直接调用相同的检查和审批。`decl.x` 本身为 `high` 风险。
- 定义保存在 `<工作目录>/.dgs-tools/definitions.json`，文件工具无法访问该位置。
  修改已有定义需要 `overwrite: true`。
- 某个步骤等待审批时，已完成的步骤会保留，之后从下一步继续。某个步骤失败时，
  该次运行会被阻止，直到宿主检查副作用并删除 `execution.memory_state["declarative_runs"]`
  中对应的记录。

## 边与路由

**条件**（`when`）：

```json
{"and": [{"eq": ["/type", "invoice"]}, {"gt": ["/amount", 1000]}, {"exists": "/customer/email"}]}
```

运算符：`eq`、`ne`、`gt`、`gte`、`lt`、`lte`、`exists`、`and`、`or`、`not`。
以 `/` 开头的字符串是指向任务数据的 JSON Pointer；需要比较字面量时写成
`{"literal": "/usr/bin"}`。读取不存在的路径会出错，并使来源 execution 失败；
可选字段请先用 `exists` 判断。

**兜底边**：`"otherwise": true` 的边只在没有普通边通过时使用。

**进入检查**：一个具名的 Python 判断函数，不写进 JSON：

```python
engine.register_entry_check("files_created", lambda context, config: ...)  # 必须返回 bool
```

边需要 `when` 条件和进入检查同时通过。名称未注册、函数抛出异常或返回非布尔值，
都会使来源 execution 失败，并且不会创建任何后继。

**遍历次数上限**：`"max_traversals": n` 限制一条边在一个任务中可被进入的次数（`0` 表示禁用）。
次数根据日志统计，保存和加载后不会清零。每条边都需要唯一的 ID；平行边请显式设置 `id`。

**路由**（`node.routing.mode`），在已通过的边中选择：

- `default`：如果给出 `default_edges` 就走这些边（空列表表示都不走）；否则走所有通过的边。
- `llm`：候选边多于一条时，由分支选择器返回边 ID 和理由；理由会写入主动记忆。
- `human`：候选边多于一条时，execution 进入等待。使用 `pending_branch_selections(task)` 和
  `resolve_branch_selection(task, request_id, edge_ids, reviewer=...)` 处理。

未被选择的边会向下游产生 `skipped` 到达，汇合节点据此知道某个分支不会再到达。
被跳过的节点不会运行处理函数、hook 或工具。

## 汇合与启动条件

没有启动条件的节点每次到达都会运行一次。启动条件会暂存到达，直到每条必需的边都有到达为止：

```json
{"id": "merge", "type": "agent.merge",
 "start_condition": {"scope": "group", "group_by": "/batch", "required_edges": ["left-to-merge", "right-to-merge"]}}
```

- `required_edges` 必须列出所有入边（默认就是全部入边）。`"join": "all"` 是任务内汇合的简写。
  汇合节点不能作为起始节点。
- `scope: "task"` 汇合同一任务的多个分支。`scope: "group"` 汇合 `group_by` 处数据值相同（且非空）的
  不同任务；它们的数据会合并到一个新任务中（不相交的键合并，相同的值保留，任何冲突都会使汇合失败
  并报告冲突路径）。来源任务变为 `merged`。
- `generation_by` 用于区分循环中的不同轮次；不设置时，每条边上的到达按先进先出配对。
- `check` 指向用 `engine.register_start_check` 注册的判断函数，它可以让汇合继续等待。
- 对任务内汇合，跳过的到达算作“已解决”；对跨任务汇合，需设置 `"accept_skipped": true` 才算。
  如果所有到达都是跳过，汇合节点本身也被跳过。

当启动检查所依赖的内容发生变化后，请调用 `engine.refresh_start_conditions(workflow)`。
任务池是单线程的协调器：使用多个 worker 的宿主必须串行化调度，并以原子方式持久化任务池。

## 前置与后置 hook

```json
{"id": "agent", "type": "agent.summarize",
 "pre_run": "inspect_folder", "pre_run_config": {"pattern": "*.txt"},
 "post_run": "write_summary", "post_run_config": {"filename": "summary.txt"}}
```

```python
engine.register_pre_run("inspect_folder", lambda context, config: {"files": [...]})
engine.register_post_run("write_summary", lambda context, result, config: None)
```

- `pre_run(context, config)` 在每次尝试中、处理函数之前运行，返回一个 JSON 对象（或 `None`），
  通过 `context.pre_run_data` 提供，也会展示给模型。
- `post_run(context, result, config)` 只在节点完成时运行，时机是输出写入之后、评估出边之前。
  它必须返回 `None`。
- 两个 hook 的名称都会在运行任何代码之前解析，因此缺少后置 hook 时不会被悄悄跳过。
- hook 出错会使 execution 失败；后置 hook 失败后可以重试，而不必重新运行处理函数（见下文）。

## 重启重放与重试

Python 的调用栈无法保存。普通处理函数或 hook 因等待人工决定而暂停后，决定到达时会从头重新运行。
为保证安全，当前阶段中每个已完成的顶层动作都会写入日志（`execution.local["action_journal"]`）。
函数重新运行时，名称和参数相同、顺序相同的调用直接从日志中返回结果：不会重复执行，
也不会再次请求审批。第一个不一致的调用结束重放，之后正常执行。ReAct 节点不需要这一机制：
其历史已经记录了每个动作，暂停的动作会按审批时的内容原样恢复执行。

`engine.retry_execution(task, execution, replay=True)` 让失败的 execution 重新变为 ready：

- 如果失败发生在后置 hook 或调度阶段，就从该阶段继续，不会重新运行处理函数；
- 否则重新运行处理函数，已完成的动作会被重放（传入 `replay=False` 则全部重新执行）；
- 存在被拒绝的审批时不能重试。

重放不等于“恰好一次”：如果进程在工具产生副作用之后、下一次保存之前崩溃，副作用仍可能重复。
在重要的地方请让外部副作用保持幂等。

## 超时

```json
{"id": "review", "type": "agent", "timeout_seconds": 120, "timeout_human_check": true}
```

截止时间从 execution 创建时开始计算，包括等待汇合和审批的时间。检查是协作式的，发生在工具调用、
模型调用和生命周期各步骤；请在调度器中定期调用 `engine.check_timeouts(workflow)`，
尤其是节点等待人工处理期间。超时后：

- `timeout_human_check: true`（默认）时创建 `timeout` 审批，批准后从批准时刻起再给一个
  `timeout_seconds` 的时限；
- 否则 execution 失败。

阻塞中的 Python、HTTP 或模型调用无法被中断；请为客户端设置各自的超时，需要强制终止时请使用进程隔离。

## 记忆

| 层次 | 存放位置 | 写入者 | 生命周期 |
| --- | --- | --- | --- |
| 客观记忆 | `task.events` | 框架和宿主 | 单个任务 |
| 主动记忆 | `task.active_memory` | 模型（需引用事件） | 单个任务 |
| 知识 | `KnowledgeStore` | 经过审核的提案 | 跨任务（见[下一节](#长期知识)） |
| 流程性记忆 | skills、声明式工具、保存的子图 | 宿主以及经审核的模型请求 | 跨任务 |

**客观记忆**就是任务日志 `task.events`，只由框架和宿主写入。每条事件包含 `id`、`timestamp`、
`sequence`、`task_id`、`execution_id`、`node_id`、`type` 和 `details`。它记录任务和节点的生命周期、
每次动作和工具调用（参数、结果、错误）、hook、边的决定、汇合、审批、安全事件、模型调用
（不含内部推理）以及 skill 活动。日志无法关闭。宿主可以用
`record_event(task, "application.note", {...})` 或 `context.record_activity(kind, details)`
追加自己的事件；浏览器可以使用 `ObservedBrowser` 或 `record_page_opened`。跨任务汇合后，
`sequence` 只在各自的来源任务内有效，不要只按它对合并后的日志排序。

**主动记忆**（`task.active_memory`）保存模型在检查点写下的笔记。每条笔记都引用客观事件 ID；
两者冲突时以日志为准。检查点在以下时机产生：节点开始、每次工具调用之后、设置文件中的轮数和
token 间隔、节点总结，以及宿主声明的名称（先在节点的 `memory_checkpoints` 中声明，再调用
`context.checkpoint("name", details)`）。没有记忆写入器时，检查点会排队等待，不会凭空生成笔记。
节点总结会把被替代的笔记移入 `active_memory["archive"]`，可用 `read_memory_archive(task, ...)` 读取。

**模型能看到什么**：提示词从不包含完整日志，而是一个受限视图：最新的事件（不含 `llm.started`
这类簿记事件）和最新的当前笔记，每部分都受字符预算限制，过长的字符串会被截短。视图会说明省略了多少内容。
ReAct 历史也以同样方式限制，记忆写入器只收到与其检查点相关的证据。相关上限见
[设置项](#设置项)；由于每部分都有预算，任务运行多久提示词都不会继续增长。

## 长期知识

客观记忆和主动记忆都属于单个任务。`KnowledgeStore` 保存需要跨任务保留的内容：事实和经验，
以**页面**的形式存放在宿主拥有的目录中，每个页面由若干条简短的**条目**组成。
它不使用向量嵌入，也不使用向量数据库。完整示例见 `python example_knowledge.py`。

```python
store = KnowledgeStore("C:/project/knowledge")        # 请放在智能体工作目录之外
engine = WorkflowEngine(tools=tools, knowledge=store, knowledge_writer=writer)
```

```json
{"id": "agent", "type": "agent", "knowledge": {"read": ["global", "support"], "propose": ["support"]}}
```

**范围（scope）**用于给页面分组（如 `global`、`support`、每个工作流一个、每个客户一个；名称使用小写）。
没有 `knowledge` 设置的节点看不到也访问不了任何知识。`propose` 中的范围必须同时出现在 `read` 中。

**读取。** 节点的提示词中包含一份自动生成的索引（页面名、标题、一行摘要），受字符预算限制；
节点同时获得三个低风险工具：`knowledge.list`、`knowledge.read` 和 `knowledge.search`
（关键词排序；中文、日文和韩文按相邻字符对匹配）。页面名为 `<scope>/<slug>`，例如 `support/refunds`。
这些工具与其他工具一样经过节点的 allowance 检查，读取会写入日志。

**写入总是间接的。** 没有任何操作会直接修改页面。一次变更由若干小操作组成：

| 操作 | 字段 |
| --- | --- |
| `create_page` | `page`、`title`、`summary` |
| `set_summary` | `page`、`summary`，可选 `title` |
| `add_entry` | `page`、`text`（一条独立完整的事实或经验） |
| `update_entry` | `page`、`entry`（ID）、`text` |
| `retire_entry` | `page`、`entry`（ID）、`reason` |

变更在提出时就会校验并试运行；其中的操作要么全部生效，要么全部不生效。生效时页面版本号加一，
变更前后的页面内容写入 `log.jsonl`。变更有三个来源：

- **整理（consolidation，主要途径）。** 当 `run_execution` 把任务结算为 completed 或 failed 时，
  `knowledge_writer` 回调会收到 `request_type: "knowledge_consolidation"` 的请求
  （任务的受限视图、索引、相关页面以及仍在等待审核的提案，以免重复提出），并返回
  `{"operations": [...], "rationale": "...", "evidence": [事件 ID]}`，或空的 `operations` 列表。
  写入器出错只会记入日志，不会让任务失败。以其他方式结束的任务，可自行调用
  `engine.consolidate_knowledge(workflow, task)`。
- **`knowledge.propose`**（中风险），供设置了 `propose` 范围的节点使用；每个任务有次数上限。
- **宿主**：`store.apply(operations, author="admin")` 立即生效。

**审核。** 提案在知识库中排队等待，不会让任务暂停：

```python
for proposal in store.proposals():                    # 待审核，按时间先后
    ...                                               # 展示操作、理由和来源
store.resolve_proposal(proposal["id"], True, reviewer="ann", comment="Matches policy")
store.revert(change_id, reviewer="ann")               # 撤销一次已生效的变更
```

每个提案都记录了所属任务、引用的日志事件，以及该任务是否使用过外部内容。只要任务运行过以下范围之外的
任何动作，就被标记为 **external**：纯数据工具（`text.*`、`json.*`、`csv.*`、`hash.sha256`、`url.build`）、
知识工具，以及宿主在 `engine.knowledge_trusted_actions` 中声明可信的动作（例如 `"tool:docs.search"`）。
声明式工具和生成的子图按它们实际运行的步骤判断。
来自 external 任务的提案始终需要人工审核。把 `knowledge_human_review` 设为 `false`，
只会让其余提案自动生效。

**维护。** `store.lint()` 报告重复条目、过大的页面、空页面、长期未读的页面和长期未处理的提案；
它不做任何修改。`store.changes()` 列出变更记录。多个进程可以共用同一个知识库目录。

**限制。** “索引放进提示词”的方式适合每个节点最多几百个页面；超过后请拆分范围，让每个节点读得更少。
搜索是关键词匹配，不是语义相似度：撰写条目时请使用人们会搜索的词。模型生成的子图继承创建它的节点的
读取范围，并且永远不能提出变更。

## Agent Skills

```python
skills = LocalToolModule()
register_skill(skills, "./skills/review", scripts=["scripts/check.py"])
engine.attach_tools(CompositeToolModule(BuiltinPrivateToolModule(workspace), skills))
```

skill 包遵循 [Agent Skills 规范](https://agentskills.io/specification)：一个目录，其中的 `SKILL.md`
带有 YAML frontmatter，`name` 与目录名一致。注册后会增加以下工具：

| 工具 | 风险 | 用途 |
| --- | --- | --- |
| `skill.<name>.invoke` | low | 打开一次调用，返回说明内容以及包中保存的图 |
| `skill.<name>.read` | low | 读取包内文件（UTF-8 或 base64） |
| `skill.<name>.finish` | low | 关闭调用 |
| `skill.<name>.script` | high，需人工审批 | 运行已登记的 `scripts/*.py`（仅在登记了脚本时提供） |
| `skill.<name>.run_graph` | high | 把包中保存的图作为经审核的子任务运行（见下文） |

这些工具与其他工具一样经过节点的 allowance 检查。调用可以通过 `parent_invoke_id` 嵌套；
父调用必须在子调用关闭后才能关闭。skill 活动会写入日志。脚本以 `python -I -` 运行，
经过校验的脚本内容从 stdin 传入，工作目录为包目录，默认超时 30 秒，输出保留前 16 KiB。
脚本属于受信任代码，不在沙箱中运行：它可以直接访问文件和网络。清单中的 `allowed-tools` 只是元数据。

## 模型生成的子图

设置了 `"allow_create_graph": true` 且拥有 `tool:graph.create` 权限（风险 `high`）的节点，
可以请求运行它自己设计的小型工作流：`nodes`、`edges`、`start` 和 `explanation`
（见 `examples/generated_graph_request.json`）。通过审核后，它会立即作为子任务运行，
初始数据是父任务数据的副本；调用返回子任务的数据。

限制：

- 每个子节点的 allowance 都必须是创建它的节点的 allowance（动作和风险上限）的子集，
  也必须是所有外层子图 allowance 的子集；省略时继承父节点的 allowance。父节点的参数检查同样生效。
  MCP server 也必须是父节点的子集。
- **宿主代码需要显式开放。** 节点类型、前置/后置 hook、进入检查和启动检查都是 Python 代码。
  子图只能使用以 `generated_graphs=True` 注册的名称，例如
  `engine.register_post_run("notify", hook, generated_graphs=True)`。
  LLM 节点类型（`register_llm`）默认允许，因为它们只能通过受检查的动作做事；
  普通的 `register` 和 `register_react` 类型默认不允许。
- 子节点继承创建它的节点的知识读取范围，并且永远不能提出知识变更。
- 子图中不支持人工分支选择。
- 嵌套深度、审批和步数预算由设置文件决定。
- 子任务中产生的审批（工具、超时）会转交给父节点的审核人。

审核时拒绝一个子图不会让创建它的节点失败：`graph.create` 调用会带着审核人的意见失败，模型可以不用这个图继续工作，
或者提出另一个图（该审核标记为 `"on_reject": "continue"`；日志记录 `graph.rejected`）。再次提出同一个图会以同样方式被拒绝。

### 保存在 skill 中的图

skill 可以保留运行时用过的图，下次运行时就可以从一个已经成功过的图开始：

1. **保存。** 存在打开的 skill 调用时，`"save_to_skill": true` 加上 Markdown 格式的 `usage_scenario`（说明何时使用该图）
   会把图保存到该 skill 的 `graphs/<id>.json` 和 `graphs/<id>.md`。保存需要单独审核，并且只在图成功运行完成后才会发起；
   运行失败的图不会被提议保存。拒绝保存不会丢弃图的运行结果（此时 `saved_path` 为 `null`）。
2. **查找。** `skill.<name>.invoke` 会返回 `graphs`：每个保存的图一条，按时间从新到旧，包含 `id`、`title`（使用场景的第一行）、
   `saved_at`、`usage_scenario`（前 1500 个字符）、`explanation` 以及节点 id 和名称。使用场景与记录的 SHA-256 不一致、
   或图本身无效的文件，会带着 `invalid` 和原因列出，而不列出其内容。
3. **复用。** 用打开的 `invoke_id` 和 `graph_id` 调用 `skill.<name>.run_graph`，会通过该包的 `SkillGraphInterface` 运行这个图。
   默认实现会加载并校验文件，然后把它作为调用节点的生成图运行：子节点的 allowance 必须是该节点的子集；
   开启 `graph_creation_human_check` 时需要审核（审核内容中会显示 `skill_graph`）；深度和步数限制同样适用。
   节点不需要 `allow_create_graph`：allowance 中的 `run_graph` 动作就是权限，而且只能运行包中已有的图。

要改变保存的图的运行方式（例如交给其他 worker），可以继承 `SkillGraphInterface`，重写
`execute_graph(context, invoke_id, graph_id)`，并把实例传给 `register_skill(..., graph_runner=runner)`。
传入 `graph_runner=None` 则不提供 `run_graph`。

`graphs/` 中任何格式相同的文件都可以这样使用，因此 skill 也可以附带手写的图。宿主可以用
`load_skill_graph(root, id)` 和 `list_skill_graphs(root)` 进行校验。

#### 运行环境与运行历史

以前成功过的图，在不同的环境下不一定还能用。框架会在每个保存的图旁边维护 `graphs/<id>.meta.json`，
并且由框架自己更新、不经过模型：图被保存时，以及每次通过 `run_graph` 运行之后（每次运行只计一次，审核暂停后恢复也不会重复计数）。

| 字段 | 内容 |
| --- | --- |
| `saved` | 保存时间以及当时的环境 |
| `runs` | `total`、`completed`、`failed` |
| `last_run` | 时间、`outcome`、`error`、节点运行次数（`steps`）、`child_task_id`、环境 |
| `last_success` | 最近一次成功运行的时间和环境 |
| `graph_sha256` | 所描述的图文件；为图的其他版本记录的元数据会被忽略 |

**环境**指图所依赖的东西：它的节点类型是否已注册、节点使用的 allowance 及其允许的工具的定义摘要、
宿主设置的 `engine.host_environment`（例如 `{"model": "anthropic/claude-opus-5-5"}` 这样的事实）以及 Python 版本。

打开 skill 时，列出的每个图还会附带：

- `history`：`saved_at`、`runs` 以及上一次运行的时间、结果和错误；
- `environment`：把当前环境与最近一次成功运行（或保存时）的环境比较得到的 `status` 和 `changes`：
  `same`；`changed`（某个工具或 allowance 的定义、或宿主事实发生了变化）；`incompatible`（某个节点类型、allowance 或工具已经不存在）；
  `unknown`（还没有运行记录）；
- `hint`：给模型的一句话，例如“The environment changed since the last successful run (model: provider/a → provider/b). Reuse it only if these changes do not matter for the task, and check its result.”

这些元数据只是参考，不受图的哈希保护。删除它只会丢失历史记录。

## 设置项

`memory_settings.json`（或 `WorkflowEngine(memory_settings_path=...)` 指定的文件）每次使用时都会重新读取。
省略的键使用默认值；未知的键会报错。

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `react_rounds` | 5 | 同一 execution 每做 N 次模型决策产生一个检查点 |
| `token_limit` | 8192 | 每累计使用这么多决策 token 产生一个检查点（通过 `MeteredDecision` 使用模型服务的计数，否则估算）。这是检查点间隔，不是费用上限。 |
| `prompt_max_events` | 100 | 展示给模型的最新日志事件数量上限 |
| `prompt_max_value_chars` | 1000 | 提示词中记忆和历史里的长字符串会被截短到此长度 |
| `prompt_budget_chars` | 30000 | 提示词各部分（事件、笔记、历史）各自的字符预算（中文、日文、韩文字符按四个计） |
| `react_full_steps` | 5 | 完整发送的最新 ReAct 步数；更早的步骤变成索引（失败的步骤始终完整保留） |
| `react_index_fold` | 10 | 每满这么多已归档步骤，折叠为一行索引（0：不折叠） |
| `prompt_knowledge_chars` | 6000 | 提示词中知识索引的字符预算 |
| `knowledge_human_review` | true | 所有知识提案都需人工审核；设为 `false` 时，未使用外部内容的任务提出的提案自动生效 |
| `knowledge_consolidate_on_finish` | true | 任务完成或失败时，向知识写入器询问值得保留的经验 |
| `knowledge_max_proposals_per_task` | 5 | 每个任务允许调用 `knowledge.propose` 的次数 |
| `graph_max_depth` | 3 | 生成子图的最大嵌套深度（0 表示禁用） |
| `graph_creation_human_check` | true | 每个生成的子图运行前需要审核 |
| `graph_save_human_check` | true | 把子图保存到 skill 前需要审核 |
| `graph_max_steps` | 100 | 每个生成的子图允许运行的节点次数 |
| `model_retry_attempts` | 0 | 模型调用暂时性失败时的重试次数（0 表示关闭） |
| `model_retry_delay_seconds` | 2 | 第一次重试前的等待秒数，之后每次加倍 |
| `model_retry_max_delay_seconds` | 30 | 两次重试之间最长的等待秒数 |

## 安全模型与限制

- **受信任代码**：处理函数、hook、进入检查和启动检查、工具后端以及 skill 脚本都是受信任的 Python 代码。
  权限只控制经过调度入口的操作，不控制这些代码直接做的事。
- **不是沙箱**：路径检查无法防御硬链接或被并发替换的文件；处理不受信任的工作时请使用操作系统级隔离。
- **提示词注入**：工具输出、记忆和知识会以“数据”的身份交给模型，并附带相应说明，但模型仍可能被误导。
  真正的边界是权限、审批和安全锁。注入的内容一旦被接受进长期知识就会长期存在，
  因此使用过外部内容的任务提出的提案始终需要人工审核。
- **并发**：引擎和任务池是同步、单线程的。使用多个 worker 的宿主必须串行化调度并以原子方式持久化；
  框架不提供分布式锁。
- **持久性**：快照以原子方式写入，但进程崩溃时，上次保存之后的事件会丢失。
- **不包含**：模型客户端、MCP 传输层、用户界面、身份认证、通知。`ToolCreationInterface`、
  `SkillWriteInterface` 和 `SkillAgentInterface` 只有接口定义。

## 与上一版本的差异

- 项目正式命名为 Directed Growing Sheet（DGS）。模块名（`workflow_*`）不变；安装包名为 `directed-growing-sheet`。
- 新的推荐用法：`python install.py`（只询问使用哪个模型，默认 Gemini 4 Hydrogen），所有默认设置都以 JSON 保存在 `presets/` 中
  （schema `assistant_preset`、`model_services`）；以及 `Assistant`、`openai_compatible` 适配器、`create_engine`、`use_default_allowance`，
  以及 `safe-default` / `workspace-editor` 两个 allowance 预设。

- 处理函数、输入、结果和调度中的错误现在会让 execution 失败，而不是抛出异常或使其卡在 `running`。
  调度错误不再让 `run_execution` 抛出异常。请使用 `engine.retry_execution`，不要手动修改状态。
- ReAct 节点把失败的动作反馈给模型，而不是直接失败；需要旧行为时设置
  `"react": {"on_action_error": "fail"}`。
- 模型生成的子图只能使用以 `generated_graphs=True` 注册的节点类型、hook 和检查（LLM 节点类型默认允许）。
- 提示词中的记忆是受限视图，由新设置 `prompt_max_events`、`prompt_max_value_chars` 和
  `prompt_budget_chars` 控制。`node.run_finished` 事件不再复制任务数据。
- 重新运行的处理函数和 hook 会重放已完成的调用，而不是重复执行。
- 拒绝审批或因超时失败后，任务状态会正确结算。
- 新增 `CompositeToolModule` 用于组合多个工具后端；skill 包的注册信息位于
  `module.skill_packages`（原为 `_skill_packages`）。
- 新增：长期知识（`KnowledgeStore`、`knowledge.*` 工具和整理流程）。
  只有挂载了知识库且节点显式开启时才会启用。
- HTTP 凭据通过 `http_credentials` 配置。
- 条件支持 `{"literal": value}`；模板目录检查不区分大小写。
- `http.request` 按服务器声明的字符集解码正文（不再总是按 UTF-8），过长的正文改为截断而不是报错，会解压 gzip/deflate，
  发送 `User-Agent`，标记拦截页，并新增用于 HTML 的 `"format": "text"` 模式。
- 新增：`engine.add_run_observer` 报告每个节点 execution（包括生成图子任务中的），用于显示进度。
- 模型调用自动重试（`model_retry_*` 设置，默认关闭；`TransientModelError`），以及按引擎覆盖设置的 `engine.setting_overrides`。
- `react.on_step_limit: "answer"` 会在 `max_steps` 用完时给节点最后一次不带工具的回合，而不是直接让它失败。
- 审核时拒绝模型生成的子图不再让创建它的节点失败；模型会收到审核人的意见并继续工作。
- 新增：较长的 ReAct 节点只完整发送最新的步骤（`react_full_steps`，5）和失败的步骤；更早的步骤变成索引
  （按 `react_index_fold` 折叠，10），模型可以用 `history.read` 读回。工具可以为索引概括自己的调用（`register(..., summary=...)`）。
  提示词预算中，中日韩字符按四个计。
- `graph.create` 向模型展示精简的 schema（生成图能用到的节点字段）；提议仍按完整 schema 校验。
- 新增：`react.result_check` 与 `engine.register_result_check` 让宿主在 ReAct 节点的结果生效前核对它的工作；
  被拒绝的结果会连同原因返回给模型。
- skill 中保存的图会由 `skill.<name>.invoke` 列出，并通过新的 `skill.<name>.run_graph` 工具运行；
  `SkillGraphInterface` 已经实现，并且可以替换。每个保存的图都会在 `graphs/<id>.meta.json` 中记录运行环境和运行历史，
  列出时会说明环境是否发生了变化（宿主通过 `engine.host_environment` 声明模型等事实）。现在保存发生在图运行完成之后，拒绝保存也不再让节点失败（`"on_reject": "continue"`）。

## 项目结构与开发

| 路径 | 内容 |
| --- | --- |
| `workflow_core.py` | 公开 API，请从这里导入 |
| `workflow_quickstart.py` | `Assistant`、`openai_compatible`、`openai_responses`、`create_engine` |
| `presets/` | 推荐用法的全部默认设置（JSON）：`allowances/`、`assistant.json`、`models.json` |
| `install.py`、`quickstart.py`、`workflow_model_setup.py` | 一步安装（含模型选择），以及终端对话 |
| `workflow_schema_compat.py` | 转换为 Gemini 接受的工具 schema 子集 |
| `GETTING_STARTED.zh-CN.md` | 面向新用户的简明指南 |
| `workflow_engine.py` | 运行、调度入口和后继安排 |
| `workflow_definitions.py`、`workflow_executions.py`、`workflow_nodes.py` | 数据模型 |
| `workflow_react.py`、`workflow_prompts.py`、`workflow_routing.py` | 智能体循环、提示词、分支选择 |
| `workflow_allowances.py`、`workflow_approvals.py`、`workflow_timeouts.py`、`workflow_replay.py` | 权限、审批、超时、重放 |
| `workflow_tools.py`、`workflow_private_tools.py`、`workflow_basic_tools.py`、`workflow_http_tools.py`、`workflow_declarative_*.py`、`workflow_tool_safety.py` | 工具 |
| `workflow_logging.py`、`workflow_memory.py`、`workflow_memory_store.py` | 日志、主动记忆、快照 |
| `workflow_knowledge.py`、`workflow_knowledge_tools.py` | 长期知识库、工具和整理 |
| `workflow_joins.py`、`workflow_start_conditions.py` | 汇合 |
| `workflow_skills.py`、`workflow_graph_tools.py`、`workflow_graphs.py`、`workflow_skill_graph_meta.py` | skill、生成图、保存的图的运行环境与历史 |
| `schemas/workflow.schema.json` | 所有定义的 JSON Schema |
| `tools/` | 内置工具目录和安全设置 |
| `examples/`、`example_*.py` | 可运行的示例 |
| `tests/` | 单元测试 |

代码格式化和静态检查使用 [ruff](https://docs.astral.sh/ruff/)，配置位于 `pyproject.toml`：

```bash
ruff format . && ruff check .
```

## 许可证

Directed Growing Sheet 采用 [Apache License 2.0](LICENSE) 许可证。
