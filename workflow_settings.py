"""Host-owned global settings shared by memory checkpoints, prompts and generated graphs.

The file is re-read on every use, so edits take effect on the next decision.
Omitted keys take the defaults below; unknown keys are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

from workflow_types import JsonDict

DEFAULT_SETTINGS = Path(__file__).with_name("memory_settings.json")

DEFAULTS: JsonDict = {
    # Write an objective checkpoint every N model decisions of one execution.
    "react_rounds": 5,
    # ... and every time this many decision tokens have been used since the last one.
    "token_limit": 8192,
    # Prompts include at most this many of the most recent objective events.
    "prompt_max_events": 100,
    # Longer strings inside prompt memory and history are cut to this many characters.
    "prompt_max_value_chars": 1000,
    # Character budget for each prompt section: events, notes and ReAct history.
    # Chinese, Japanese and Korean characters count four times (about a token each).
    "prompt_budget_chars": 30000,
    # ReAct prompts carry this many of a node's newest steps in full; older steps
    # become an index the model can read back with history.read (failed steps
    # always stay in full). Per node: react.full_steps.
    "react_full_steps": 5,
    # Fold each complete block of this many archived steps into one index line
    # (0: one line per step). Per node: react.index_fold.
    "react_index_fold": 10,
    # Character budget for the knowledge page index shown to a node.
    "prompt_knowledge_chars": 6000,
    # Every knowledge proposal waits for a person. When false, proposals from tasks
    # that used no outside content are applied automatically.
    "knowledge_human_review": True,
    # Ask the knowledge writer for lessons when a task completes or fails.
    "knowledge_consolidate_on_finish": True,
    # Proposals a task's nodes may make with the knowledge.propose tool.
    "knowledge_max_proposals_per_task": 5,
    "graph_max_depth": 3,
    "graph_creation_human_check": True,
    "graph_save_human_check": True,
    "graph_max_steps": 100,
    # Retry a model call that timed out, lost its connection or found the provider
    # overloaded, this many times (0 switches retries off; see workflow_model_retry).
    "model_retry_attempts": 0,
    # Wait before the first retry, doubled for each further one, up to the maximum.
    "model_retry_delay_seconds": 2,
    "model_retry_max_delay_seconds": 30,
}

_MINIMUM = {"graph_max_depth": 0, "model_retry_attempts": 0, "react_index_fold": 0}


def load_settings(
    path: str | Path = DEFAULT_SETTINGS, overrides: JsonDict | None = None
) -> JsonDict:
    """Settings from the file, then ``overrides`` (a host's per-engine values)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("Global settings must be a JSON object.")
    raw = {**raw, **(overrides or {})}
    unknown = raw.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown global settings: {sorted(unknown)}")
    settings = {**DEFAULTS, **raw}
    for name, value in settings.items():
        if isinstance(DEFAULTS[name], bool):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a Boolean.")
        elif type(value) is not int or value < _MINIMUM.get(name, 1):
            raise ValueError(f"{name} must be an integer >= {_MINIMUM.get(name, 1)}.")
    return settings
