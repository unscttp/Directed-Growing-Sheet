from __future__ import annotations

import copy
import uuid
from typing import Any

from workflow_types import JsonDict


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _split_pointer(pointer: str) -> list[str]:
    """Minimal JSON Pointer support."""
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer!r}")
    parts = pointer[1:].split("/")
    return [part.replace("~1", "/").replace("~0", "~") for part in parts]


def get_value(data: Any, pointer: str) -> Any:
    current = data
    for part in _split_pointer(pointer):
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise KeyError(f"Cannot read {pointer!r}; {part!r} is not reachable.")
    return current


def set_value(data: JsonDict, pointer: str, value: Any) -> None:
    parts = _split_pointer(pointer)

    if not parts:
        if not isinstance(value, dict):
            raise ValueError("Replacing Task.data root requires a JSON object.")
        data.clear()
        data.update(copy.deepcopy(value))
        return

    current: Any = data

    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
            continue

        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]

    last = parts[-1]

    if isinstance(current, list):
        index = int(last)
        if index == len(current):
            current.append(copy.deepcopy(value))
        else:
            current[index] = copy.deepcopy(value)
    elif isinstance(current, dict):
        current[last] = copy.deepcopy(value)
    else:
        raise KeyError(f"Cannot write {pointer!r}.")
