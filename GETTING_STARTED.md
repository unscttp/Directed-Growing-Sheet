**English** | [简体中文](GETTING_STARTED.zh-CN.md)

# Getting started with Directed Growing Sheet

Directed Growing Sheet (DGS) lets an AI model work for you step by step: it
reads your files, works with text and data, and answers. Every step it takes is
checked against what you allow, and recorded in the task's journal.

This guide gets you from download to your first answer in about five minutes.
The only thing you decide is which AI model to use. You don't need to know how
the engine works; when you want the details, read [README.md](README.md).

## What you need

- **Python 3.10 or newer.** Check with `python --version` (on some systems
  `python3 --version`). Download it from [python.org](https://www.python.org/downloads/)
  if needed; on Windows, tick "Add python.exe to PATH" during installation.
- **An API key for the model you want**: a Google AI Studio key for Gemini
  ([aistudio.google.com](https://aistudio.google.com)), a Claude Console key
  for Claude, an OpenAI key for GPT or a DeepSeek key. A model running on your
  own computer with Ollama or LM Studio needs no key.

## 1. Install and choose the model

Download this folder (with `git clone` or as a ZIP from the repository page),
open a terminal in it, and run:

```bash
python install.py
```

It installs everything into a private Python environment (the `.venv` folder),
runs a short self-check, and then asks which model to use:

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

Press Enter for Gemini 4 Hydrogen, or type a number. Then paste your API key
(it is not shown while you type) and confirm the model. The installer checks
which models your key can use, offers a list if needed, and sends one small
test request. Your choice is saved in `.dgs/config.json` on your computer; the
key is not sent anywhere except to the model's service.

A warning about an "editable installation" is normal.

## 2. Switch to the environment

macOS and Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Do this again whenever you open a new terminal.

## 3. Chat

```bash
python quickstart.py
```

Type a request and press Enter; an empty line quits. The assistant works in a
`workspace` folder in the current directory (created automatically; choose
another with `--workspace path`). Put files there and ask about them, for
example:

- "What files are in the workspace?"
- "Summarise report.txt in three bullet points."
- "How many rows in sales.csv have a total above 100?"

To use another model later:

```bash
python quickstart.py --setup
```

## What it may do

By default the assistant uses the **safe default** permissions: it can read
files inside the workspace and work with text, JSON and CSV. It cannot write or
delete files, use the internet or install anything, and it cannot reach files
outside the workspace.

To let it write files in the workspace (for example "save the summary to
summary.md"), start it with:

```bash
python quickstart.py --allow-writes
```

If something ever needs your approval, it asks in the terminal before acting.

## Use it from Python

```python
from workflow_core import Assistant, openai_compatible

model = openai_compatible(
    "gemini-4-hydrogen",
    api_key="your-key",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)
assistant = Assistant(model, workspace="workspace")
print(assistant.ask("Summarise notes.txt").answer)
```

## If something goes wrong

| Message | What to do |
| --- | --- |
| `python: command not found` | Try `python3`, or install Python (see above). |
| PowerShell: "running scripts is disabled" | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` in Command Prompt. |
| "The key was not accepted" | Copy the key again from your provider; check that it is enabled for the API. |
| "... is not among the models this key can use" | Pick one of the listed models. |
| "No model chosen yet" | Run `python quickstart.py --setup`. |
| Timeouts or "HTTP 429" | The service is busy or you hit a limit. DGS retries a few times by itself; try again later. |
| It says it cannot do something | It may need a permission it does not have, for example `--allow-writes`. |

## Next steps

- The model list, the assistant's instructions and its permissions are plain
  JSON files in `presets/` (`models.json`, `assistant.json`, `allowances/`).
  Edit them to add a model or change what the assistant may do.
- [README.md](README.md) explains workflows with several steps, your own tools,
  approvals, long-term memory, skills and more.
- `example_*.py` are small runnable examples.
