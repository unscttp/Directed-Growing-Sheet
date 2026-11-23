[English](GETTING_STARTED.md) | **简体中文**

# Directed Growing Sheet 入门

Directed Growing Sheet（DGS）让 AI 模型一步一步地替你完成工作：读取你的文件、处理文本和数据，然后给出回答。
它的每一步都会按照你允许的范围进行检查，并记录在任务日志中。

按照本指南，大约五分钟就能从下载到得到第一个回答。你唯一需要决定的是使用哪个 AI 模型。
你不需要了解引擎的工作原理；需要详细信息时，请阅读 [README.zh-CN.md](README.zh-CN.md)。

## 需要准备

- **Python 3.10 或更高版本。** 用 `python --version`（有些系统上是 `python3 --version`）检查。
  如果没有，请从 [python.org](https://www.python.org/downloads/) 下载；在 Windows 上安装时请勾选“Add python.exe to PATH”。
- **所选模型的 API Key**：Gemini 使用 Google AI Studio 密钥（[aistudio.google.com](https://aistudio.google.com)），
  Claude 使用 Claude Console 密钥，GPT 使用 OpenAI 密钥，或者 DeepSeek 密钥。用 Ollama 或 LM Studio 在自己电脑上运行的模型不需要密钥。

## 1. 安装并选择模型

下载本目录（用 `git clone`，或在仓库页面下载 ZIP），在该目录中打开终端，运行：

```bash
python install.py
```

它会把所需的一切安装到独立的 Python 环境（`.venv` 目录）中，运行一个简短的自检，然后询问使用哪个模型：

```
Which model should the assistant use?
  1) Gemini 4 Hydrogen (Google)   (default, press Enter)
  2) Claude Opus 5.5 (Anthropic)
  3) GPT-6.2 Astra (OpenAI)
  4) DeepSeek V4 Pro
  5) Ollama on this computer
  6) LM Studio on this computer
  7) Another OpenAI-compatible API
```

直接按回车选择 Gemini 4 Hydrogen，或输入编号。然后粘贴 API Key（输入时不会显示），并确认模型。
安装程序会检查这个密钥可以使用哪些模型，必要时列出供你选择，并发送一次很小的测试请求。
你的选择保存在本机的 `.dgs/config.json` 中；密钥只会发送给模型服务，不会发往其他地方。

出现关于“editable installation”的警告是正常的。

## 2. 进入该环境

macOS 和 Linux：

```bash
source .venv/bin/activate
```

Windows（PowerShell）：

```powershell
.venv\Scripts\Activate.ps1
```

每次打开新的终端都需要重新执行这一步。

## 3. 开始对话

```bash
python quickstart.py
```

输入请求后按回车；输入空行即退出。助手在当前目录下的 `workspace` 目录中工作（会自动创建；可以用 `--workspace 路径` 指定其他目录）。
把文件放进去，然后向它提问，例如：

- “工作目录里有哪些文件？”
- “用三条要点总结 report.txt。”
- “sales.csv 里总额超过 100 的有多少行？”

以后想换用其他模型：

```bash
python quickstart.py --setup
```

## 它能做什么

默认情况下，助手使用**安全默认**权限：可以读取工作目录中的文件，处理文本、JSON 和 CSV。
它不能写入或删除文件，不能访问互联网或安装任何东西，也无法访问工作目录以外的文件。

如果想让它在工作目录中写文件（例如“把总结保存到 summary.md”），用下面的方式启动：

```bash
python quickstart.py --allow-writes
```

如果某个操作需要你的批准，它会先在终端中询问你。

## 在 Python 中使用

```python
from workflow_core import Assistant, openai_compatible

model = openai_compatible(
    "gemini-4-hydrogen",
    api_key="你的密钥",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)
assistant = Assistant(model, workspace="workspace")
print(assistant.ask("总结 notes.txt").answer)
```

## 遇到问题时

| 提示 | 处理方法 |
| --- | --- |
| `python: command not found` | 试试 `python3`，或安装 Python（见上文）。 |
| PowerShell 提示“禁止运行脚本” | 执行一次 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，或在命令提示符中使用 `.venv\Scripts\activate.bat`。 |
| “The key was not accepted” | 从服务商处重新复制密钥，并确认它已开通 API 权限。 |
| “... is not among the models this key can use” | 从列出的模型中选择一个。 |
| “No model chosen yet” | 运行 `python quickstart.py --setup`。 |
| 超时或“HTTP 429” | 服务繁忙或达到了限额。DGS 会自动重试几次；稍后再试。 |
| 它说无法完成某件事 | 可能缺少相应权限，例如需要 `--allow-writes`。 |

## 下一步

- 模型列表、助手的说明和权限都是 `presets/` 中的普通 JSON 文件（`models.json`、`assistant.json`、`allowances/`）。
  修改这些文件就可以添加模型，或调整助手能做的事情。
- [README.zh-CN.md](README.zh-CN.md) 介绍了多步骤工作流、自定义工具、审批、长期记忆、技能等更多内容。
- `example_*.py` 是一些可以直接运行的小示例。
