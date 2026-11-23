"""Small deterministic JSON expression language for edge ``when`` conditions.

Examples::

    {"eq": ["/classification/type", "invoice"]}
    {"gt": ["/invoice/amount", 1000]}
    {"exists": "/customer/email"}
    {"eq": ["/path", {"literal": "/usr/bin"}]}
    {"and": [expr1, expr2]}, {"or": [expr1, expr2]}, {"not": expr}

In comparisons, a string starting with ``/`` is a JSON Pointer into task data.
Wrap a value as ``{"literal": value}`` to compare against it as-is. Reading a
path that does not exist is an error (use ``exists`` to test for it first).
"""

from __future__ import annotations

import operator
from typing import Any

from workflow_types import JsonDict
from workflow_utils import get_value

COMPARISONS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
}


def _operand(value: Any, data: JsonDict) -> Any:
    if isinstance(value, dict) and set(value) == {"literal"}:
        return value["literal"]
    if isinstance(value, str) and value.startswith("/"):
        try:
            return get_value(data, value)
        except (KeyError, IndexError, ValueError) as exc:
            raise KeyError(
                f"Condition path {value!r} does not exist in task data; "
                'guard it with {"exists": ...} or compare against {"literal": ...}.'
            ) from exc
    return value


def evaluate_condition(expr: Any, data: JsonDict) -> bool:
    if isinstance(expr, bool):
        return expr
    if not isinstance(expr, dict) or len(expr) != 1:
        raise ValueError(f"Invalid condition: {expr!r}")
    op, raw = next(iter(expr.items()))

    if op == "and":
        return all(evaluate_condition(item, data) for item in raw)
    if op == "or":
        return any(evaluate_condition(item, data) for item in raw)
    if op == "not":
        return not evaluate_condition(raw, data)
    if op == "exists":
        try:
            get_value(data, raw)
            return True
        except (KeyError, IndexError, ValueError):
            return False

    if op not in COMPARISONS:
        raise ValueError(f"Unknown condition operator: {op!r}")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"Operator {op!r} expects two values.")
    left, right = (_operand(value, data) for value in raw)
    return bool(COMPARISONS[op](left, right))
