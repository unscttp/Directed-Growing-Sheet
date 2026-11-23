"""Replay of completed actions when trusted Python code restarts after a pause.

A Python call stack cannot be persisted. When an ordinary handler or hook pauses
for a human decision, the engine reruns that function from the top once the
decision arrives. Without help, every action it completed before the pause would
run again, and every approval it consumed would be requested again.

``ActionReplay`` keeps an ordered journal of the top-level actions that completed
during one phase of one execution. When the function reruns, the n-th call is
answered from the journal if it has the same action name and arguments as the
n-th journal entry. The first call that differs ends replay: the remaining
journal entries are discarded and execution continues normally.

ReAct loops do not need this: their history already persists every action.
Nested calls (for example the steps of a declarative tool) are not journaled,
because the outer call owns their resumption.
"""

from __future__ import annotations

import copy
from typing import Any

from workflow_types import JsonDict

JOURNAL_KEY = "action_journal"


class ActionReplay:
    def __init__(self, journal: list[JsonDict], *, enabled: bool = True) -> None:
        self.journal = journal
        self.enabled = enabled
        self.cursor = 0
        self.depth = 0

    @classmethod
    def for_phase(cls, execution, phase: str) -> ActionReplay:
        journals = execution.local.setdefault(JOURNAL_KEY, {})
        return cls(journals.setdefault(phase, []))

    @staticmethod
    def clear(execution, phase: str | None = None) -> None:
        journals = execution.local.get(JOURNAL_KEY)
        if journals is None:
            return
        if phase is None:
            execution.local.pop(JOURNAL_KEY, None)
        else:
            journals.pop(phase, None)
            if not journals:
                execution.local.pop(JOURNAL_KEY, None)

    @property
    def active(self) -> bool:
        """Only top-level calls of an enabled phase take part in replay."""
        return self.enabled and self.depth == 0

    def lookup(self, action: str, arguments: JsonDict) -> tuple[bool, Any, bool]:
        """Return ``(hit, result, diverged)`` for the next top-level call."""
        if not self.active or self.cursor >= len(self.journal):
            return False, None, False
        entry = self.journal[self.cursor]
        if entry["action"] == action and entry["arguments"] == arguments:
            self.cursor += 1
            return True, copy.deepcopy(entry["result"]), False
        del self.journal[self.cursor :]
        return False, None, True

    def record(self, action: str, arguments: JsonDict, result: Any) -> None:
        if not self.active:
            return
        self.journal.append(
            {
                "action": action,
                "arguments": copy.deepcopy(arguments),
                "result": copy.deepcopy(result),
            }
        )
        self.cursor = len(self.journal)
