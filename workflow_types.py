"""Shared type aliases."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from workflow_nodes import NodeContext, NodeResult

JsonDict = dict[str, Any]
NodeHandler = Callable[["NodeContext"], "NodeResult"]
