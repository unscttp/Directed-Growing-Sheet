"""Reusable JSON allowance sets. No assigned set means unrestricted actions."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from workflow_risk import risk_rank
from workflow_schema import validate_document
from workflow_types import JsonDict


@dataclass
class AllowanceSet:
    id: str
    actions: list[str] = field(default_factory=list)
    description: str = ""
    config: JsonDict = field(default_factory=dict)
    max_tool_risk: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Allowance id must be a nonempty string.")
        if not isinstance(self.actions, list) or any(
            not isinstance(action, str) or not action.strip() for action in self.actions
        ):
            raise ValueError("Allowance actions must be a list of nonempty names.")
        if not isinstance(self.config, dict):
            raise ValueError("Allowance config must be a JSON object.")
        if self.max_tool_risk is not None:
            risk_rank(self.max_tool_risk)

    def assert_subset_of(self, parent):
        if parent is None:
            return
        if not set(self.actions) <= set(parent.actions):
            raise PermissionError("Child graph allowance exceeds parent actions.")
        if parent.max_tool_risk is not None and (
            self.max_tool_risk is None or not parent.allows_tool_risk(self.max_tool_risk)
        ):
            raise PermissionError("Child graph risk allowance exceeds parent.")

    def allows_tool_risk(self, risk_level: str) -> bool:
        rank = risk_rank(risk_level)
        return self.max_tool_risk is None or rank <= risk_rank(self.max_tool_risk)

    def allows(self, action: str) -> bool:
        """Exact action names; an empty list allows no actions."""
        return action in self.actions

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: JsonDict) -> AllowanceSet:

        validate_document(raw, "allowance")
        return cls(
            id=raw["id"],
            actions=copy.deepcopy(raw.get("actions", [])),
            description=raw.get("description", ""),
            config=copy.deepcopy(raw.get("config", {})),
            max_tool_risk=raw.get("max_tool_risk"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> AllowanceSet:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class AllowanceRegistry:
    """Application-owned sets; register again to replace a user's edited set."""

    def __init__(self) -> None:
        self.sets: dict[str, AllowanceSet] = {}

    def register(self, allowance: AllowanceSet) -> None:
        validated = AllowanceSet.from_dict(allowance.to_dict())
        self.sets[validated.id] = validated

    def load(self, path: str | Path) -> AllowanceSet:
        allowance = AllowanceSet.load(path)
        self.register(allowance)
        return allowance

    def resolve(self, reference: str | None) -> AllowanceSet | None:
        if reference is None:
            return None
        if reference not in self.sets:
            raise KeyError(f"Unknown allowance set: {reference!r}")
        return copy.deepcopy(self.sets[reference])
