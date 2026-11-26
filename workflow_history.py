"""A ReAct node's step history in prompts: recent steps in full, older ones archived.

Every step a node takes (its action and the observation) is kept in full in the
execution's journal (``TaskExecution.local["react"]``). Sending all of them
with every model request is what makes long nodes expensive, so prompts carry:

* the newest ``react_full_steps`` steps (default 5) in full;
* every failed step anywhere in the node (an error observation or a result the
  node's check rejected) in full, because errors are what the model must not
  repeat;
* the rest as an index: one numbered line per step ("#7 library.decide - 25
  decisions"), and once a block of ``react_index_fold`` steps (default 10) is
  complete, one line for the whole block ("#1-#10: called a x5, b x5").

The model reads archived steps again with the ``history.read`` tool, by step
number. Index lines never change once written and blocks close at fixed
boundaries, so the start of the prompt stays stable between requests and
provider prompt caches keep working.

A tool can describe its own calls for the index: ``LocalToolModule.register(...,
summary=fn)`` with ``fn(arguments, result) -> str``. Other tools get a generic
line built from their arguments and result.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable

from workflow_tools import ToolDefinition
from workflow_types import JsonDict

HISTORY_TOOL = "history.read"
MAX_READ_STEPS = 10
STEPS = re.compile(r"^\s*\d+(\s*-\s*\d+)?(\s*,\s*\d+(\s*-\s*\d+)?)*\s*$")
INDEX_NOTE = (
    "Earlier steps of this node, archived to save space. Their full records are "
    'kept: read any of them with history_read (step numbers such as "7", "3,9" '
    'or "11-20"). Failed steps are not archived; they appear in full below.'
)

Summary = Callable[[str, JsonDict, object], "str | None"]


def is_error(entry: JsonDict) -> bool:
    """A failed step: an error observation, or a result the node's check rejected."""
    if "action" not in entry:
        return True
    observation = entry.get("observation")
    return isinstance(observation, dict) and set(observation) == {"error"}


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _value(value) -> str:
    if isinstance(value, str):
        return _clip(value, 40)
    if isinstance(value, list):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        return "{…}"
    return json.dumps(value)


def _generic(arguments: JsonDict, result) -> str:
    args = ", ".join(f"{k}={_value(v)}" for k, v in list((arguments or {}).items())[:2])
    if isinstance(result, dict):
        fields = [
            f"{k}={_value(v)}"
            for k, v in result.items()
            if not isinstance(v, (dict, list)) or isinstance(v, list)
        ][:3]
        outcome = ", ".join(fields) or "ok"
    elif isinstance(result, list):
        outcome = f"{len(result)} items"
    elif result is None:
        outcome = "ok"
    else:
        outcome = _clip(str(result), 60)
    return f"({args}) → {outcome}" if args else f"→ {outcome}"


def _name(entry: JsonDict) -> str:
    return str(entry["action"]["name"]).removeprefix("tool:")


def step_line(number: int, entry: JsonDict, summarize: Summary | None = None) -> str:
    """The index line of one archived step."""
    if "action" not in entry:
        return f"#{number} answer not accepted (in full below)"
    name = _name(entry)
    if is_error(entry):
        return f"#{number} {name} failed (in full below)"
    text = None
    if summarize is not None:
        try:
            text = summarize(name, entry["action"].get("arguments") or {}, entry["observation"])
        except Exception:  # a broken summary must never break the prompt
            text = None
    detail = (
        _clip(text, 160)
        if text
        else _generic(entry["action"].get("arguments") or {}, entry.get("observation"))
    )
    return f"#{number} {name} - {detail}"


def fold_line(start: int, entries: list[JsonDict]) -> str:
    """One line for a complete block of archived steps."""
    counts: dict[str, int] = {}
    failed = []
    for offset, entry in enumerate(entries):
        name = _name(entry) if "action" in entry else "answer"
        counts[name] = counts.get(name, 0) + 1
        if is_error(entry):
            failed.append(f"#{start + offset}")
    calls = ", ".join(f"{name} ×{count}" for name, count in counts.items())
    line = f"#{start}-#{start + len(entries) - 1}: called {calls}"
    if failed:
        line += f"; failed: {', '.join(failed)} (in full below)"
    return line


def archive(
    history: list[JsonDict], full_steps: int, fold: int, summarize: Summary | None = None
) -> tuple[list[str], list[int]]:
    """``(index lines, positions of the steps sent in full)`` for a node's history.

    ``full_steps`` newest steps and every failed step are sent in full; the
    others become index lines, a complete block of ``fold`` steps one line
    (``fold`` 0: never fold).
    """
    recent = max(0, len(history) - full_steps)
    lines: list[str] = []
    position = 0
    if fold > 0:
        while position + fold <= recent:
            lines.append(fold_line(position + 1, history[position : position + fold]))
            position += fold
    lines.extend(step_line(i + 1, history[i], summarize) for i in range(position, recent))
    kept = [i for i in range(recent) if is_error(history[i])] + list(range(recent, len(history)))
    return lines, kept


def parse_steps(spec: str, available: int) -> list[int]:
    """Step numbers from "7", "3,9" or "11-20", within the node's history."""
    if not isinstance(spec, str) or not STEPS.match(spec):
        raise ValueError('steps looks like "7", "3,9" or "11-20".')
    numbers: list[int] = []
    for part in spec.split(","):
        first, _, last = part.strip().partition("-")
        start, end = int(first), int(last or first)
        if start < 1 or end < start:
            raise ValueError(f"Invalid step range {part.strip()!r}.")
        numbers.extend(range(start, end + 1))
    numbers = sorted(set(numbers))
    if len(numbers) > MAX_READ_STEPS:
        raise ValueError(f"Read at most {MAX_READ_STEPS} steps per call.")
    missing = [n for n in numbers if n > available]
    if missing:
        raise ValueError(f"This node has {available} earlier steps; no step {missing[0]}.")
    return numbers


class HistoryTools:
    """The built-in ``history.read`` tool: a node reads its own archived steps.

    It needs no allowance entry: it only shows the node what the node already
    did and saw. It is offered once a node has archived steps.
    """

    def get_tool(self, name: str) -> ToolDefinition:
        if name != HISTORY_TOOL:
            raise KeyError(name)
        return ToolDefinition(
            HISTORY_TOOL,
            "low",
            description=(
                "Read earlier steps of this node in full (the action and what it returned), "
                f'by the numbers in the step index: "7", "3,9" or "11-20"; up to '
                f"{MAX_READ_STEPS} steps per call."
            ),
            input_schema={
                "type": "object",
                "properties": {"steps": {"type": "string", "pattern": STEPS.pattern}},
                "required": ["steps"],
                "additionalProperties": False,
            },
        )

    def execute(self, context, tool: ToolDefinition, arguments: JsonDict):
        history = context.execution.local.get("react", [])
        numbers = parse_steps(arguments.get("steps"), len(history))
        steps = []
        for number in numbers:
            entry = {k: v for k, v in history[number - 1].items() if k != "model_state"}
            steps.append({"step": number, **copy.deepcopy(entry)})
        return {"steps": steps}
