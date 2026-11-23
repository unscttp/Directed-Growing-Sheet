"""Interface-only extension points for MCP and private tool implementations.

These are synchronous application adapters compatible with ToolModule, not an
MCP transport implementation. No SDK, network connection, or private dispatcher
is provided here. Attach private tools with attach_tools and MCP sessions with attach_mcp_server.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from workflow_nodes import NodeContext
from workflow_tools import ToolDefinition, require_tool_human_check
from workflow_types import JsonDict


class ToolBackendInterface(ABC):
    """Shared contract; the engine retains allowance, risk, and approval checks."""

    def authorize_call(
        self, context: NodeContext, tool: ToolDefinition, arguments: JsonDict
    ) -> None:
        """Run after engine allowance checks, before execute, without tool effects.

        Trusted adapters may override to block unsafe calls and notify their UI.
        Overrides must retain require_tool_human_check (or equivalent persistent
        human gating). Raise to prevent execution; never interpret a UI callback
        return value as approval. Legacy backends without this hook keep the
        engine's default human gate.
        """
        require_tool_human_check(context, tool, arguments)

    @abstractmethod
    def list_tools(self) -> list[ToolDefinition]:
        """Describe available tools, with explicit application-assigned risk levels.

        Discovery must not execute tools. Return detached metadata so callers
        cannot accidentally mutate the implementation's catalog.
        """
        raise NotImplementedError

    @abstractmethod
    def get_tool(self, name: str) -> ToolDefinition:
        """Return the named tool's metadata without execution; unknown names raise KeyError."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, context: NodeContext, tool: ToolDefinition, arguments: JsonDict) -> Any:
        """Execute the exact tool described by the engine-approved definition.

        The implementation must reject a changed definition, return a
        JSON-compatible result, and raise on failure. It must not silently
        dispatch to a different tool or treat remote errors as successful output.
        The engine calls this only after permission and human-review checks.
        """
        raise NotImplementedError


class MCPToolModule(ToolBackendInterface):
    """Extension point for an MCP client adapter, independent of SDK/transport.

    Implementations map remote tool descriptions to local ToolDefinition objects
    and normalize remote results in execute. Transport, session management,
    discovery/pagination, and authentication belong to the future implementation.
    The host explicitly manages connection lifecycle; attach_mcp_server does not connect.
    """

    @abstractmethod
    def connect(self, config: JsonDict) -> None:
        """Initialize the chosen MCP connection using implementation-defined settings."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release connection/session resources; repeated close calls should be safe."""
        raise NotImplementedError


class PrivateToolModule(ToolBackendInterface):
    """Extension point for user-owned dispatchers, services, or business logic.

    Subclasses supply discovery, metadata lookup, and execution through the same
    engine boundary as MCP tools. No private-tool execution strategy is prescribed.
    """

    @abstractmethod
    def configure(self, config: JsonDict) -> None:
        """Prepare user-defined dispatch settings before attaching to the engine."""
        raise NotImplementedError


class ToolCreationInterface(ABC):
    """Reserved extension point for future LLM-requested tool creation.

    No implementation, code execution, registration, or agent-visible action is
    installed. The host will define implementation_request and the creation,
    validation, authorization, and persistence policy in a later stage.
    """

    @abstractmethod
    def add_tool(
        self,
        context: NodeContext,
        description: ToolDefinition,
        implementation_request: JsonDict,
    ) -> ToolDefinition:
        """Future contract: return the resulting definition or raise on failure."""
        raise NotImplementedError
