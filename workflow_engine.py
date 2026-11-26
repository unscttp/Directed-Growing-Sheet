"""The workflow interpreter: runs node executions and schedules their successors.

One call to ``run_execution`` runs one ready ``TaskExecution`` synchronously:

    pre-run hook -> handler (or ReAct loop) -> apply outputs -> post-run hook
    -> optional post-run human review -> edge selection and scheduling

Every action a node takes goes through ``perform_action``, which enforces the
node's allowance, tool risk ceilings, generated-graph ceilings, tool safety
policies and human approvals, and records the call in the task journal.

Persistence is deliberately outside the engine: the host saves ``Task`` or
``TaskPool`` snapshots when it chooses.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

from workflow_allowances import AllowanceRegistry, AllowanceSet
from workflow_approvals import HumanCheckRequired, approval_server
from workflow_conditions import evaluate_condition
from workflow_definitions import EdgeDefinition, NodeDefinition, WorkflowDefinition
from workflow_edge_checks import EdgeCheckContext, EntryCheck
from workflow_executions import Task, TaskExecution
from workflow_graph_tools import GraphTools
from workflow_history import HISTORY_TOOL, HistoryTools
from workflow_hooks import PostRunHook, PreRunHook, require_post_run_human_check
from workflow_joins import TaskPool
from workflow_knowledge import KnowledgeStore
from workflow_knowledge_tools import TOOL_PREFIX as KNOWLEDGE_TOOL_PREFIX
from workflow_knowledge_tools import KnowledgeTools
from workflow_logging import audit_execution, record_event, snapshot
from workflow_memory import drain_checkpoints, write_checkpoint
from workflow_nodes import NodeContext, NodeResult
from workflow_prompts import (
    LLMDecision,
    LLMReasoner,
    node_prompt_context,
    tool_prompt_descriptions,
)
from workflow_react import ObserveHandler, ReActHandler, ReasonHandler
from workflow_replay import ActionReplay
from workflow_routing import select_edges, traversal_count
from workflow_settings import load_settings
from workflow_start_conditions import StartConditionCheck
from workflow_timeouts import arm_timeout, check_timeout
from workflow_tool_interfaces import MCPToolModule
from workflow_tools import ToolDefinition, ToolModule, require_tool_human_check
from workflow_types import JsonDict, NodeHandler
from workflow_utils import get_value, set_value

NODE_RESULT_STATUSES = ("completed", "waiting", "failed")
RESERVED_TOOL_NAMES = ("graph.create", HISTORY_TOOL)

# Kinds of host code a model-generated graph may reference only when the host
# registered it with ``generated_graphs=True``.
GENERATED_GRAPH_KINDS = (
    "handler",
    "pre_run",
    "post_run",
    "entry_check",
    "start_check",
    "result_check",
)

AllowanceCheck = Callable[[NodeContext, AllowanceSet, str, JsonDict], bool]


def _require_name(name: str, what: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{what} name must be a nonempty string.")


def _require_callable(function, what: str) -> None:
    if not callable(function):
        raise TypeError(f"{what} must be callable.")


class WorkflowEngine:
    """Interpreter for the JSON workflow model.

    Register a handler for each node type, for example
    ``engine.register("form.approval", handler)`` or
    ``engine.register_llm("agent", model_adapter)``.
    """

    def __init__(
        self,
        allowances: AllowanceRegistry | None = None,
        allowance_check: AllowanceCheck | None = None,
        tools: ToolModule | None = None,
        task_pool: TaskPool | None = None,
        memory_settings_path: str | Path | None = None,
        knowledge: KnowledgeStore | None = None,
        knowledge_writer: Callable | None = None,
    ) -> None:
        self.allowances = allowances if allowances is not None else AllowanceRegistry()
        self.allowance_check = allowance_check
        self.task_pool = task_pool if task_pool is not None else TaskPool()
        self.memory_settings_path = (
            Path(memory_settings_path)
            if memory_settings_path
            else Path(__file__).with_name("memory_settings.json")
        )
        self.handlers: dict[str, NodeHandler] = {}
        self.actions: dict[str, Callable[[NodeContext, JsonDict], Any]] = {}
        self.entry_checks: dict[str, EntryCheck] = {}
        self.start_checks: dict[str, StartConditionCheck] = {}
        self.pre_run_hooks: dict[str, PreRunHook] = {}
        self.post_run_hooks: dict[str, PostRunHook] = {}
        self.memory_writers: dict[str, Callable] = {}
        # Checks of a ReAct node's final result, named by node.react["result_check"].
        self.result_checks: dict[str, Callable[[NodeContext, NodeResult], str | None]] = {}
        self.branch_selectors: dict[str, Callable] = {}
        self.generated_graph_names: dict[str, set[str]] = {
            kind: set() for kind in GENERATED_GRAPH_KINDS
        }
        self.mcp_servers: dict[str, MCPToolModule] = {}
        self.graph_tools = GraphTools(self)
        self.history_tools = HistoryTools()
        self.knowledge_tools = KnowledgeTools(self)
        self.knowledge: KnowledgeStore | None = None
        self.knowledge_writer: Callable | None = None
        # Actions (such as "tool:docs.search") whose results the host vouches for;
        # tasks that only ran these still count as having no outside content.
        self.knowledge_trusted_actions: set[str] = set()
        self.attach_knowledge(knowledge)
        if knowledge_writer is not None:
            self.register_knowledge_writer(knowledge_writer)
        # (creating node context, allowance snapshot) for each enclosing generated graph.
        self.graph_permission_ceilings: list[tuple[NodeContext, JsonDict | None]] = []
        # Shared with the engines of generated graphs (they are shallow copies).
        self.run_observers: list[Callable[[Task, TaskExecution], None]] = []
        # Facts about the host that affect how saved skill graphs behave, for example
        # {"model": "anthropic/claude-opus-5-5"}. Recorded with each run of such a
        # graph and compared when the skill is opened (workflow_skill_graph_meta).
        self.host_environment: JsonDict = {}
        # Values that replace the settings file for this engine only, for example
        # {"model_retry_attempts": 3} to switch automatic model retries on.
        self.setting_overrides: JsonDict = {}
        self.attach_tools(tools)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _mark_generated(self, kind: str, name: str, allowed: bool) -> None:
        if type(allowed) is not bool:
            raise TypeError("generated_graphs must be a Boolean.")
        names = self.generated_graph_names[kind]
        if allowed:
            names.add(name)
        else:
            names.discard(name)

    def register(
        self, node_type: str, handler: NodeHandler, *, generated_graphs: bool = False
    ) -> None:
        """Register or replace the handler for a node type.

        ``generated_graphs=True`` lets model-generated graphs create nodes of
        this type. Leave it off for handlers whose Python code does anything an
        allowance cannot restrict.
        """
        _require_name(node_type, "Node type")
        _require_callable(handler, "Node handler")
        self.handlers[node_type] = handler
        self._mark_generated("handler", node_type, generated_graphs)

    def register_react(
        self,
        node_type: str,
        reason: ReasonHandler,
        observe: ObserveHandler | None = None,
        *,
        generated_graphs: bool = False,
    ) -> None:
        self.register(node_type, ReActHandler(reason, observe), generated_graphs=generated_graphs)

    def register_llm(
        self,
        node_type: str,
        llm: LLMDecision,
        *,
        memory_writer: Callable | None | Literal[False] = None,
        branch_selector: Callable | None = None,
        generated_graphs: bool = True,
    ) -> None:
        """Run nodes of this type as a ReAct loop driven by a model adapter.

        Without separate callbacks the same adapter also answers memory and
        branch-selection requests (distinguished by ``request_type``). Pass
        ``memory_writer=False`` to write no active-memory notes for this node
        type; objective checkpoints are still recorded. LLM nodes act only
        through allowance-checked actions, so generated graphs may use them by
        default.
        """
        self.register_react(node_type, LLMReasoner(llm), generated_graphs=generated_graphs)
        if memory_writer is False:
            self.memory_writers.pop(node_type, None)
        else:
            self.memory_writers[node_type] = memory_writer or llm
        self.branch_selectors[node_type] = branch_selector or llm

    def register_memory_writer(self, node_type: str, writer: Callable) -> None:
        _require_callable(writer, "Memory writer")
        self.memory_writers[node_type] = writer

    def register_branch_selector(self, node_type: str, selector: Callable) -> None:
        _require_callable(selector, "Branch selector")
        self.branch_selectors[node_type] = selector

    def register_action(self, name: str, handler: Callable[[NodeContext, JsonDict], Any]) -> None:
        _require_name(name, "Action")
        _require_callable(handler, "Action handler")
        if name.startswith("tool:"):
            raise ValueError("The 'tool:' prefix is reserved for the attached tool module.")
        self.actions[name] = handler

    def register_pre_run(
        self, name: str, hook: PreRunHook, *, generated_graphs: bool = False
    ) -> None:
        """Register or replace a preparation function referenced by ``node.pre_run``."""
        _require_name(name, "Pre-run hook")
        _require_callable(hook, "Pre-run hook")
        self.pre_run_hooks[name] = hook
        self._mark_generated("pre_run", name, generated_graphs)

    def register_post_run(
        self, name: str, hook: PostRunHook, *, generated_graphs: bool = False
    ) -> None:
        """Register or replace a completion function referenced by ``node.post_run``."""
        _require_name(name, "Post-run hook")
        _require_callable(hook, "Post-run hook")
        self.post_run_hooks[name] = hook
        self._mark_generated("post_run", name, generated_graphs)

    def register_result_check(
        self,
        name: str,
        check: Callable[[NodeContext, NodeResult], str | None],
        *,
        generated_graphs: bool = False,
    ) -> None:
        """Register a check of ReAct results, referenced by ``node.react.result_check``.

        ``check(context, result)`` runs when the model gives its final result and
        returns None to accept it, or an explanation to reject it: the result is
        then not applied, the explanation goes back to the model as an
        observation and the loop goes on (each rejection uses a step). If the step
        limit is reached without an accepted result, the node fails with the
        explanation. Use it to verify that a node did its work (files written,
        records complete) instead of trusting its summary.
        """
        _require_name(name, "Result check")
        _require_callable(check, "Result check")
        self.result_checks[name] = check
        self._mark_generated("result_check", name, generated_graphs)

    def add_run_observer(self, observer: Callable[[Task, TaskExecution], None]) -> None:
        """Call ``observer(task, execution)`` before every node execution runs.

        Executions of generated child graphs are included, so a host can follow
        a child task while the parent's tool call is still running, for example
        to show progress. Observers must not change the task; an observer that
        raises is recorded as ``observer.failed`` and the run continues.
        """
        _require_callable(observer, "Run observer")
        self.run_observers.append(observer)

    def register_entry_check(
        self, name: str, check: EntryCheck, *, generated_graphs: bool = False
    ) -> None:
        """Register or replace a Boolean edge predicate referenced by ``edge.entry_check``."""
        _require_name(name, "Entry check")
        _require_callable(check, "Entry check")
        self.entry_checks[name] = check
        self._mark_generated("entry_check", name, generated_graphs)

    def register_start_check(
        self, name: str, check: StartConditionCheck, *, generated_graphs: bool = False
    ) -> None:
        """Register or replace a join predicate referenced by ``start_condition.check``."""
        if not isinstance(name, str) or not name.strip() or not callable(check):
            raise ValueError("A node start check requires a nonempty name and a callable.")
        self.start_checks[name] = check
        self._mark_generated("start_check", name, generated_graphs)

    def attach_knowledge(self, store: KnowledgeStore | None) -> None:
        """Attach or detach the long-term knowledge store."""
        if store is not None and not isinstance(store, KnowledgeStore):
            raise TypeError("Expected a KnowledgeStore.")
        self.knowledge = store

    def register_knowledge_writer(self, writer: Callable) -> None:
        """Model callback asked what a finished task should add to knowledge."""
        _require_callable(writer, "Knowledge writer")
        self.knowledge_writer = writer

    def attach_tools(self, module: ToolModule | None) -> None:
        """Attach, replace or detach the local tool backend without changing nodes."""
        if isinstance(module, MCPToolModule):
            raise ValueError("Use attach_mcp_server(server_id, adapter) for MCP tools.")
        if module is not None and (
            not callable(getattr(module, "get_tool", None))
            or not callable(getattr(module, "execute", None))
        ):
            raise TypeError("Tool module must implement get_tool and execute.")
        self.tools = module

    def attach_mcp_server(self, server_id: str, adapter: MCPToolModule | None) -> None:
        """Bind a host-managed MCP session to a stable ID used in node definitions."""
        if not isinstance(server_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", server_id):
            raise ValueError("Invalid MCP server ID.")
        if adapter is None:
            self.mcp_servers.pop(server_id, None)
        elif isinstance(adapter, MCPToolModule):
            self.mcp_servers[server_id] = adapter
        else:
            raise TypeError("Expected an MCPToolModule adapter.")

    def _required_servers(self, node: NodeDefinition) -> dict[str, MCPToolModule]:
        servers = {}
        for server in node.requires.get("mcp_servers", []):
            if server not in self.mcp_servers:
                raise RuntimeError(f"Required MCP server {server!r} is not attached.")
            servers[server] = self.mcp_servers[server]
        return servers

    # ------------------------------------------------------------------
    # Settings, contexts and memory
    # ------------------------------------------------------------------

    def settings(self) -> JsonDict:
        """Re-read the global settings file, so edits apply to the next decision."""
        return load_settings(self.memory_settings_path, self.setting_overrides)

    def build_inputs(self, node: NodeDefinition, task: Task) -> JsonDict:
        return {
            name: copy.deepcopy(get_value(task.data, pointer))
            for name, pointer in node.inputs.items()
        }

    def build_context(
        self,
        workflow: WorkflowDefinition,
        task: Task,
        execution: TaskExecution,
        *,
        inputs: JsonDict | None = None,
        pre_run_data: JsonDict | None = None,
    ) -> NodeContext:
        node = workflow.node(execution.node_id)
        return NodeContext(
            workflow=workflow,
            node=node,
            task=task,
            execution=execution,
            inputs=self.build_inputs(node, task) if inputs is None else copy.deepcopy(inputs),
            pre_run_data=copy.deepcopy(pre_run_data or {}),
            runtime=self,
        )

    def _bare_context(self, workflow, task, execution) -> NodeContext:
        """A context for engine bookkeeping that does not need node inputs."""
        return self.build_context(workflow, task, execution, inputs={})

    def checkpoint(self, context, name, details=None, *, key=None, kind="note"):
        writer = self.memory_writers.get(context.node.type)
        if key is None:
            drain_checkpoints(context, writer)
        return write_checkpoint(context, writer, name, details, key=key, kind=kind)

    def record_react_round(self, context: NodeContext) -> None:
        """Count a model decision; queue checkpoints at the configured intervals."""
        settings = self.settings()
        state = context.execution.memory_state
        state["react_rounds"] = state.get("react_rounds", 0) + 1
        for counter, setting in (
            ("react_rounds", "react_rounds"),
            ("decision_tokens", "token_limit"),
        ):
            total = state.get(counter, 0)
            baseline = "checkpoint_" + counter
            if total - state.get(baseline, 0) >= settings[setting]:
                write_checkpoint(
                    context,
                    None,
                    setting,
                    {"total": total, "limit": settings[setting]},
                    key=f"{setting}:{total}",
                )
                state[baseline] = total

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def graph_ceiling_policies(self):
        """Yield (creating context, policy) for every enclosing generated graph."""
        for parent_context, allowance_snapshot in self.graph_permission_ceilings:
            policies = [self.allowances.resolve(parent_context.node.allowance)]
            if allowance_snapshot is not None:
                policies.append(AllowanceSet.from_dict(allowance_snapshot))
            for policy in policies:
                if policy is not None:
                    yield parent_context, policy

    def _tool_catalog(self, context: NodeContext, settings: JsonDict) -> tuple[list, bool]:
        discover = getattr(self.tools, "list_tools", None)
        catalog = list(discover()) if callable(discover) else []
        for tool in catalog:
            if isinstance(tool, ToolDefinition) and (
                tool.name.startswith(("mcp:", KNOWLEDGE_TOOL_PREFIX))
                or tool.name in RESERVED_TOOL_NAMES
            ):
                raise ValueError(
                    "Local tool names cannot use reserved MCP, graph or knowledge tool names."
                )
        depth = context.task.metadata.get("_graph_depth", 0)
        if context.node.allow_create_graph and depth < settings["graph_max_depth"]:
            catalog.append(self.graph_tools.get_tool("graph.create"))
        catalog.extend(self.knowledge_tools.visible_tools(context.node))
        for server, adapter in self._required_servers(context.node).items():
            for tool in adapter.list_tools():
                if not isinstance(tool, ToolDefinition):
                    raise TypeError("MCP discovery must return ToolDefinition objects.")
                catalog.append(replace(tool, name=f"mcp:{server}:{tool.name}"))
        for _, ceiling in self.graph_ceiling_policies():
            catalog = [
                tool
                for tool in catalog
                if ceiling.allows("tool:" + tool.name) and ceiling.allows_tool_risk(tool.risk_level)
            ]
        return catalog, callable(discover) or bool(catalog)

    def node_prompt_context(self, context: NodeContext) -> JsonDict:
        """Everything a model needs to decide the next step of this node."""
        settings = self.settings()
        allowance = self.allowances.resolve(context.node.allowance)
        prompt = node_prompt_context(context, allowance)
        prompt["graph_runtime"] = {
            "depth": context.task.metadata.get("_graph_depth", 0),
            "max_depth": settings["graph_max_depth"],
            "creation_human_check": settings["graph_creation_human_check"],
            "save_human_check": settings["graph_save_human_check"],
        }
        knowledge = self.knowledge_tools.prompt_view(context.node, settings)
        if knowledge is not None:
            prompt["knowledge"] = knowledge
        catalog, available = self._tool_catalog(context, settings)
        prompt["tool_catalog_available"] = available
        prompt["available_tools"] = tool_prompt_descriptions(
            catalog, allowance, argument_check_required=self.allowance_check is not None
        )
        # Offered once some of the node's steps are archived (see workflow_history).
        full_steps = context.node.react.get("full_steps", settings["react_full_steps"])
        if len(context.execution.local.get("react") or []) > full_steps:
            prompt["available_tools"].extend(
                tool_prompt_descriptions([self.history_tools.get_tool(HISTORY_TOOL)], None)
            )
        return prompt

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def perform_action(self, context: NodeContext, action: str, arguments: JsonDict) -> Any:
        """Check, run and journal one action or tool call for a node."""
        kind = "tool" if isinstance(action, str) and action.startswith("tool:") else "action"
        task, execution = context.task, context.execution
        replay = context.replay
        if replay is not None and replay.active:
            hit, result, diverged = replay.lookup(action, snapshot(arguments))
            if diverged:
                record_event(task, "replay.diverged", {"action": snapshot(action)}, execution)
            if hit:
                record_event(
                    task,
                    f"{kind}.replayed",
                    {
                        "action": action,
                        "arguments": snapshot(arguments),
                        "result": snapshot(result),
                    },
                    execution,
                )
                return result

        call_id = record_event(
            task,
            f"{kind}.requested",
            {
                "action": snapshot(action),
                "arguments": snapshot(arguments),
                "allowance": context.node.allowance,
            },
            execution,
        )
        if replay is not None:
            replay.depth += 1
        try:
            check_timeout(context)
            with audit_execution(task, execution):
                result = self._dispatch(context, action, arguments, call_id)
        except HumanCheckRequired as exc:
            record_event(
                task,
                f"{kind}.waiting",
                {"call_id": call_id, "request_id": exc.request_id},
                execution,
            )
            if kind == "tool":
                write_checkpoint(context, None, "tool_waiting", {"call_id": call_id}, key=call_id)
            raise
        except Exception as exc:
            record_event(
                task,
                f"{kind}.failed",
                {"call_id": call_id, "error_type": type(exc).__name__, "error": str(exc)},
                execution,
            )
            if kind == "tool":
                write_checkpoint(
                    context,
                    None,
                    "tool_failed",
                    {"call_id": call_id, "error": str(exc)},
                    key=call_id,
                )
            raise
        finally:
            if replay is not None:
                replay.depth -= 1

        record_event(
            task, f"{kind}.completed", {"call_id": call_id, "result": snapshot(result)}, execution
        )
        if replay is not None:
            replay.record(action, snapshot(arguments), result)
        if kind == "tool":
            # Only queued here. ReAct commits the observation before the model writer
            # runs, so a writer failure never replays a successful tool.
            write_checkpoint(context, None, "tool", {"call_id": call_id}, key=call_id)
        return result

    def _resolve_tool(self, context: NodeContext, action: str):
        """Return (definition, backend, MCP server ID) for a ``tool:`` action."""
        name = action[len("tool:") :]
        module, server = self.tools, None
        if name == HISTORY_TOOL:
            module = self.history_tools
        elif name in RESERVED_TOOL_NAMES:
            module = self.graph_tools
        elif name.startswith(KNOWLEDGE_TOOL_PREFIX):
            module = self.knowledge_tools
        elif name.startswith("mcp:"):
            parts = name.split(":", 2)
            if len(parts) != 3:
                raise ValueError("MCP tool actions look like 'tool:mcp:<server>:<tool>'.")
            _, server, name = parts
            servers = self._required_servers(context.node)
            if server not in servers:
                raise PermissionError(f"MCP server {server!r} is not declared by this node.")
            module = servers[server]
        if module is None:
            raise RuntimeError("No tool module is attached.")
        tool = module.get_tool(name)
        if not isinstance(tool, ToolDefinition):
            raise TypeError("Tool module must return a ToolDefinition.")
        tool = ToolDefinition.from_dict(tool.to_dict())
        if tool.name != name:
            raise ValueError("Tool module returned a different tool name.")
        return tool, module, server

    def _check_permissions(self, context, action, tool, arguments) -> None:
        allowance = self.allowances.resolve(context.node.allowance)
        if allowance is not None:
            if not allowance.allows(action):
                raise PermissionError(f"Action {action!r} is not allowed by {allowance.id!r}.")
            if tool is not None and not allowance.allows_tool_risk(tool.risk_level):
                raise PermissionError(
                    f"Tool {tool.name!r} risk {tool.risk_level!r} exceeds allowance "
                    f"{allowance.id!r} maximum {allowance.max_tool_risk!r}."
                )
            if self.allowance_check is not None and not self.allowance_check(
                context, allowance, action, copy.deepcopy(arguments)
            ):
                raise PermissionError(f"Action {action!r} was rejected by the allowance check.")
        for parent_context, ceiling in self.graph_ceiling_policies():
            if not ceiling.allows(action) or (
                tool is not None and not ceiling.allows_tool_risk(tool.risk_level)
            ):
                raise PermissionError("Child action exceeds its creating node's allowance.")
            if self.allowance_check is not None and not self.allowance_check(
                parent_context, ceiling, action, copy.deepcopy(arguments)
            ):
                raise PermissionError("Child action violates its creating node's argument policy.")

    def _dispatch(self, context: NodeContext, action: str, arguments: JsonDict, call_id: str):
        if not isinstance(action, str) or not action:
            raise ValueError("Action name must be a nonempty string.")
        if not isinstance(arguments, dict):
            raise ValueError("Action arguments must be a JSON object.")
        json.dumps(arguments, allow_nan=False)
        arguments = copy.deepcopy(arguments)
        is_tool = action.startswith("tool:")
        tool, module, server = (
            self._resolve_tool(context, action) if is_tool else (None, None, None)
        )
        if not (tool is not None and tool.name == HISTORY_TOOL):
            # history.read only shows a node its own earlier steps: no allowance needed.
            self._check_permissions(context, action, tool, arguments)

        if tool is None:
            if action not in self.actions:
                raise KeyError(f"Unknown action: {action!r}")
            record_event(
                context.task,
                "action.executing",
                {"call_id": call_id, "action": action, "arguments": arguments},
                context.execution,
            )
            return self.actions[action](context, arguments)

        authorize = getattr(module, "authorize_call", None)
        with approval_server(server):
            if authorize is not None:
                authorize(context, tool, arguments)
            else:
                require_tool_human_check(context, tool, arguments)
        record_event(
            context.task,
            "tool.executing",
            {"call_id": call_id, "action": action, "tool": tool.to_dict(), "arguments": arguments},
            context.execution,
        )
        result = module.execute(context, tool, arguments)
        json.dumps(result, allow_nan=False)
        return result

    # ------------------------------------------------------------------
    # Running executions
    # ------------------------------------------------------------------

    @staticmethod
    def validate_result(node: NodeDefinition, result: Any) -> None:
        """Reject a handler result that cannot be applied, before anything is applied."""
        if not isinstance(result, NodeResult):
            raise TypeError(f"Node handler must return NodeResult, got {type(result).__name__}.")
        if result.status not in NODE_RESULT_STATUSES:
            raise ValueError(f"Node result status must be one of {NODE_RESULT_STATUSES}.")
        if not isinstance(result.values, dict) or not isinstance(result.local, dict):
            raise TypeError("NodeResult values and local must be JSON objects.")
        undeclared = set(result.values) - set(node.outputs)
        if result.status == "completed" and undeclared:
            raise ValueError(f"Node {node.id!r} returned undeclared outputs: {sorted(undeclared)}")
        json.dumps({"values": result.values, "local": result.local}, allow_nan=False)

    def apply_result(
        self, task: Task, node: NodeDefinition, execution: TaskExecution, result: NodeResult
    ) -> None:
        """Commit a validated result to the execution and, when completed, to task data."""
        self.validate_result(node, result)
        record_event(task, "node.result", {"result": asdict(result)}, execution)
        execution.local.update(copy.deepcopy(result.local))
        if result.status == "failed":
            execution.status = "failed"
            execution.error = result.error or "Node failed."
        elif result.status == "waiting":
            execution.status = "waiting"
        else:
            for output_name, value in result.values.items():
                set_value(task.data, node.outputs[output_name], value)
            execution.output_data = copy.deepcopy(result.values)
            execution.status = "completed"
            execution.error = None
        task.revision += 1

    def run_execution(
        self, workflow: WorkflowDefinition, task: Task, execution: TaskExecution
    ) -> NodeResult:
        """Run one ready execution and schedule its successors when it completes.

        Handler, hook and scheduling errors become a failed result; only misuse
        (an execution that is not ready, an unregistered node type) raises.
        """
        record_event(
            task,
            "node.run_requested",
            {"status": execution.status, "attempt": execution.attempt + 1},
            execution,
        )
        for observer in self.run_observers:
            try:
                observer(task, execution)
            except Exception as exc:
                record_event(
                    task,
                    "observer.failed",
                    {"error_type": type(exc).__name__, "error": str(exc)},
                    execution,
                )
        try:
            with audit_execution(task, execution):
                result = self._run_execution(workflow, task, execution)
        except Exception as exc:
            record_event(
                task,
                "node.run_error",
                {"error_type": type(exc).__name__, "error": str(exc)},
                execution,
            )
            raise
        if result.status in {"failed", "waiting"}:
            self._finalize_memory(workflow, task, execution, result)
        if result.status in {"completed", "failed"}:
            record_event(task, "node.exited", {"status": result.status}, execution)
        record_event(
            task,
            "node.run_finished",
            {"status": execution.status, "result": asdict(result)},
            execution,
        )
        task.settle_status("no active executions remain")
        if task.status in {"completed", "failed"}:
            self.knowledge_tools.on_task_finished(workflow, task)
        return result

    def consolidate_knowledge(
        self, workflow: WorkflowDefinition, task: Task, *, force: bool = False
    ) -> JsonDict | None:
        """Propose what a finished task should add to long-term knowledge.

        Runs automatically when ``run_execution`` settles a task. Call it yourself
        for tasks that end another way (a rejected approval, a timeout), or with
        ``force=True`` to consolidate again. Returns the proposal, if any.
        """
        return self.knowledge_tools.consolidate(workflow, task, force=force)

    def retry_execution(self, task: Task, execution: TaskExecution, *, replay: bool = True) -> None:
        """Make a failed execution ready to run again.

        A run that failed in its post-run hook or while scheduling resumes from
        that phase; the successful handler is not run again. Otherwise the
        handler reruns. With ``replay=True`` actions that already completed are
        answered from the execution's journal instead of running twice.
        """
        if execution not in task.executions:
            raise ValueError("The execution does not belong to this task.")
        if execution.status != "failed":
            raise ValueError(f"Only failed executions can be retried, not {execution.status!r}.")
        if any(request["status"] == "rejected" for request in execution.human_checks):
            raise PermissionError("A human rejected this execution; it cannot be retried.")
        if not replay:
            ActionReplay.clear(execution)
        execution.status, execution.error = "ready", None
        task.revision += 1
        record_event(task, "node.retry_requested", {"replay": replay}, execution)
        if task.status in {"failed", "completed"}:
            task.set_status("running", "execution retried")

    def _finalize_memory(self, workflow, task, execution, result: NodeResult) -> None:
        context = self._bare_context(workflow, task, execution)
        writer = self.memory_writers.get(context.node.type)
        try:
            drain_checkpoints(context, writer)
            if result.status == "failed":
                write_checkpoint(
                    context,
                    writer,
                    "node_summary",
                    {"status": "failed", "error": result.error},
                    key=f"node_summary_failed:{execution.attempt}",
                    kind="summary",
                )
        except Exception as exc:
            record_event(task, "memory.finalization_pending", {"error": str(exc)}, execution)

    def _run_execution(
        self, workflow: WorkflowDefinition, task: Task, execution: TaskExecution
    ) -> NodeResult:
        if execution.status != "ready":
            raise ValueError(f"Execution {execution.id!r} is not ready: {execution.status!r}")
        node = workflow.node(execution.node_id)
        workflow.assert_valid()
        arm_timeout(node, execution)
        self.task_pool.add(task)

        rule = workflow.start_rule(node.id)
        if rule is not None and not execution.joined_from:
            execution.status = "waiting_start"
            task.revision += 1
            record_event(task, "node.waiting_start", {"start_condition": rule}, execution)
            return NodeResult(status="waiting")
        if rule is not None and (
            {receipt["edge_id"] for receipt in execution.joined_from} != set(rule["required_edges"])
            or any(receipt["rule"] != rule for receipt in execution.joined_from)
        ):
            failed = NodeResult(
                status="failed", error="Start condition changed after join release."
            )
            self.apply_result(task, node, execution, failed)
            return failed

        if execution.continuation:
            state = execution.continuation
            context = self.build_context(
                workflow,
                task,
                execution,
                inputs=state["inputs"],
                pre_run_data=state["pre_run_data"],
            )
            return self._finish_execution(context)

        if node.type not in self.handlers:
            raise KeyError(f"No handler registered for node type {node.type!r}")
        execution.status = "running"
        execution.attempt += 1
        record_event(task, "node.started", {"attempt": execution.attempt}, execution)
        context = self.build_context(workflow, task, execution, inputs={})
        context.replay = ActionReplay.for_phase(execution, "run")
        try:
            result = self._run_handler(context)
        except HumanCheckRequired:
            result = NodeResult(status="waiting")
        except Exception as exc:
            result = NodeResult(status="failed", error=f"{type(exc).__name__}: {exc}")
        self.apply_result(task, node, execution, result)
        if execution.status != "completed":
            return result

        ActionReplay.clear(execution, "run")
        execution.continuation = {
            "phase": "post_run",
            "result": asdict(result),
            "inputs": copy.deepcopy(context.inputs),
            "pre_run_data": copy.deepcopy(context.pre_run_data),
        }
        return self._finish_execution(context)

    def _run_handler(self, context: NodeContext) -> NodeResult:
        """Pre-run hook and handler. Raises on failure; the caller converts it."""
        node, execution = context.node, context.execution
        context.inputs = self.build_inputs(node, context.task)
        execution.input_data = copy.deepcopy(context.inputs)
        check_timeout(context)
        # Reject missing references before any preparation or agent side effects.
        self._required_servers(node)
        self.allowances.resolve(node.allowance)
        if node.pre_run is not None and node.pre_run not in self.pre_run_hooks:
            raise KeyError(f"Unknown pre-run hook: {node.pre_run!r}")
        if node.post_run is not None and node.post_run not in self.post_run_hooks:
            raise KeyError(f"Unknown post-run hook: {node.post_run!r}")

        if node.pre_run is not None:
            information = self._run_hook(
                context,
                "pre_run",
                node.pre_run,
                self.pre_run_hooks[node.pre_run],
                copy.deepcopy(node.pre_run_config),
            )
            if information is None:
                information = {}
            if not isinstance(information, dict):
                raise TypeError("Pre-run hook must return a JSON object or None.")
            json.dumps(information, allow_nan=False)
            context.pre_run_data = copy.deepcopy(information)
            execution.local["pre_run"] = copy.deepcopy(information)
            execution.input_data = copy.deepcopy(context.inputs)

        self.checkpoint(context, "node_start", key="node_start")
        context.drain_memory()
        check_timeout(context)
        handler = self.handlers[node.type]
        # ReAct history already persists every action; replay is for plain Python.
        context.replay.enabled = not isinstance(handler, ReActHandler)
        result = handler(context)
        self.validate_result(node, result)
        return result

    def _run_hook(self, context: NodeContext, phase: str, name: str, hook, *args):
        check_timeout(context)
        details = {"phase": phase, "name": name}
        record_event(context.task, "hook.started", details, context.execution)
        try:
            result = hook(context, *args)
        except Exception as exc:
            record_event(
                context.task,
                "hook.waiting" if isinstance(exc, HumanCheckRequired) else "hook.failed",
                {**details, "error_type": type(exc).__name__, "error": str(exc)},
                context.execution,
            )
            raise
        record_event(
            context.task,
            "hook.completed",
            {**details, "result": snapshot(result)},
            context.execution,
        )
        return result

    def _finish_execution(self, context: NodeContext) -> NodeResult:
        """Post-run hook, post-run review and scheduling, resumable phase by phase."""
        workflow, task, execution = context.workflow, context.task, context.execution
        state = execution.continuation
        result = NodeResult(**copy.deepcopy(state["result"]))
        context.replay = ActionReplay.for_phase(execution, "post_run")
        try:
            check_timeout(context)
            if state["phase"] == "post_run":
                self._run_post_run(context, result)
                ActionReplay.clear(execution, "post_run")
                state["phase"] = "edges"
            check_timeout(context)
            if not state.get("post_reviewed"):
                require_post_run_human_check(context)
                state["post_reviewed"] = True
        except HumanCheckRequired:
            waiting = NodeResult(status="waiting")
            self.apply_result(task, context.node, execution, waiting)
            return waiting
        except Exception as exc:
            stage = "Post-run hook" if state["phase"] == "post_run" else "Post-run review"
            failed = NodeResult(status="failed", error=f"{stage}: {type(exc).__name__}: {exc}")
            self.apply_result(task, context.node, execution, failed)
            return failed

        if execution.status != "completed":
            execution.status, execution.error = "completed", None
            task.revision += 1
        try:
            self.schedule_after(workflow, task, execution)
        except HumanCheckRequired:
            execution.status = "waiting"
            return NodeResult(status="waiting")
        except Exception as exc:
            # The node's outputs stay committed; retry_execution resumes scheduling.
            error = f"Scheduling: {type(exc).__name__}: {exc}"
            execution.status, execution.error = "failed", error
            task.revision += 1
            record_event(
                task,
                "node.scheduling_failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
                execution,
            )
            return NodeResult(status="failed", error=error)
        execution.continuation = {}
        return result

    def _run_post_run(self, context: NodeContext, result: NodeResult) -> None:
        name = context.node.post_run
        if name is None:
            return
        if name not in self.post_run_hooks:
            raise KeyError(f"Unknown post-run hook: {name!r}")
        returned = self._run_hook(
            context,
            "post_run",
            name,
            self.post_run_hooks[name],
            copy.deepcopy(result),
            copy.deepcopy(context.node.post_run_config),
        )
        if returned is not None:
            raise TypeError("Post-run hook must return None.")

    # ------------------------------------------------------------------
    # Edges and scheduling
    # ------------------------------------------------------------------

    def can_enter_edge(
        self, workflow: WorkflowDefinition, task: Task, edge: EdgeDefinition
    ) -> bool:
        details = {"edge_id": workflow.edge_key(edge), "check": edge.entry_check}
        try:
            allowed = self._can_enter_edge(workflow, task, edge)
        except Exception as exc:
            record_event(
                task,
                "edge.check_failed",
                {**details, "error": str(exc), "error_type": type(exc).__name__},
            )
            raise
        record_event(task, "edge.checked", {**details, "allowed": allowed})
        return allowed

    def _can_enter_edge(self, workflow, task, edge) -> bool:
        edge_id = workflow.edge_key(edge)
        if (
            edge.max_traversals is not None
            and traversal_count(task, edge_id) >= edge.max_traversals
        ):
            record_event(
                task,
                "edge.budget_exhausted",
                {"edge_id": edge_id, "max_traversals": edge.max_traversals},
            )
            return False
        if edge.entry_check is None:
            return True
        if edge.entry_check not in self.entry_checks:
            raise KeyError(f"Unknown entry check: {edge.entry_check!r}")
        result = self.entry_checks[edge.entry_check](
            EdgeCheckContext(workflow, task, edge), copy.deepcopy(edge.entry_check_config)
        )
        if type(result) is not bool:
            raise TypeError(f"Entry check {edge.entry_check!r} must return a Boolean.")
        return result

    def choose_next_nodes(
        self, workflow: WorkflowDefinition, task: Task, node_id: str
    ) -> list[str]:
        return [edge.to_node for edge in self.choose_next_edges(workflow, task, node_id)]

    def choose_next_edges(
        self, workflow: WorkflowDefinition, task: Task, node_id: str
    ) -> list[EdgeDefinition]:
        """Ordinary edges whose condition and entry check pass, else passing fallbacks."""
        matched: list[EdgeDefinition] = []
        fallback: list[EdgeDefinition] = []
        for edge in workflow.outgoing(node_id):
            if edge.otherwise:
                fallback.append(edge)
            elif edge.when is None or evaluate_condition(edge.when, task.data):
                if self.can_enter_edge(workflow, task, edge):
                    matched.append(edge)
            else:
                record_event(
                    task,
                    "edge.skipped",
                    {"edge_id": workflow.edge_key(edge), "reason": "when condition is false"},
                )
        if matched:
            return matched
        return [edge for edge in fallback if self.can_enter_edge(workflow, task, edge)]

    def schedule_after(
        self, workflow: WorkflowDefinition, task: Task, execution: TaskExecution
    ) -> list[TaskExecution]:
        """Create successor executions for a completed execution.

        Every route, branch choice and join correlation value is computed
        before anything is recorded, so an error leaves no partial arrivals.
        """
        if execution.status != "completed":
            raise ValueError("Only a completed execution may schedule outgoing edges.")
        if execution.scheduled:
            return []
        self.task_pool.add(task)
        context = self._bare_context(workflow, task, execution)
        check_timeout(context)
        writer = self.memory_writers.get(context.node.type)
        drain_checkpoints(context, writer)
        eligible = self.choose_next_edges(workflow, task, execution.node_id)
        chosen = select_edges(context, eligible, self.branch_selectors.get(context.node.type))
        check_timeout(context)
        self.checkpoint(context, "node_summary", key="node_summary", kind="summary")
        check_timeout(context)

        chosen_ids = {workflow.edge_key(edge) for edge in chosen}
        plans = [
            (edge, self.task_pool.arrival_record(workflow, task, execution, edge))
            for edge in workflow.outgoing(execution.node_id)
        ]
        skip_plans = {
            workflow.edge_key(edge): self._skip_plan(workflow, task, execution, edge)
            for edge, _ in plans
            if workflow.edge_key(edge) not in chosen_ids
        }

        created: list[TaskExecution] = []
        for edge, receipt in plans:
            edge_id = workflow.edge_key(edge)
            if edge_id not in chosen_ids:
                self._propagate_skip(
                    workflow, task, execution, first_edge=edge, plan=skip_plans[edge_id]
                )
                continue
            successor = TaskExecution.create(edge.to_node)
            self.task_pool.stamp(receipt)
            successor.arrival = receipt
            if receipt["rule"] is not None:
                successor.status = "waiting_start"
            task.executions.append(successor)
            arm_timeout(workflow.node(successor.node_id), successor)
            record_event(
                task,
                "edge.entered",
                {
                    "edge_id": edge_id,
                    "source_execution_id": execution.id,
                    "status": successor.status,
                },
                successor,
            )
            record_event(
                task, "edge.exited", {"edge_id": edge_id, "destination": edge.to_node}, successor
            )
            created.append(successor)
        execution.scheduled = True
        self.refresh_start_conditions(workflow)
        task.settle_status("all paths resolved")
        return created

    def _skip_plan(self, workflow, task, source, first_edge=None):
        """Plan skipped arrivals; each edge is visited at most once per path."""
        if first_edge is not None:
            queue = [(source, first_edge)]
        else:
            queue = [(source, edge) for edge in workflow.outgoing(source.node_id)]
        plan = []
        while queue:
            previous, edge = queue.pop(0)
            key = workflow.edge_key(edge)
            if key in previous.skip_path:
                continue
            receipt = self.task_pool.arrival_record(workflow, task, previous, edge)
            receipt["resolution"] = "skipped"
            child = TaskExecution.create(edge.to_node)
            child.skip_path = previous.skip_path + [key]
            child.arrival = receipt
            rule = receipt["rule"]
            stops_at_group_join = rule is not None and not rule["accept_skipped"]
            if rule is not None and not stops_at_group_join:
                child.status = "waiting_start"
            else:
                child.status = "skipped"
                child.scheduled = True
                if not stops_at_group_join:
                    queue.extend(
                        (child, following) for following in workflow.outgoing(child.node_id)
                    )
            plan.append((previous, edge, child))
        return plan

    def _propagate_skip(self, workflow, task, source, first_edge=None, plan=None) -> None:
        if plan is None:
            plan = self._skip_plan(workflow, task, source, first_edge)
        for previous, edge, child in plan:
            self.task_pool.stamp(child.arrival)
            arm_timeout(workflow.node(child.node_id), child)
            task.executions.append(child)
            record_event(
                task,
                "edge.skipped",
                {
                    "edge_id": workflow.edge_key(edge),
                    "source_execution_id": previous.id,
                    "resolution": "skipped",
                },
                child,
            )
        if first_edge is None:
            source.scheduled = True

    def refresh_start_conditions(self, workflow: WorkflowDefinition):
        """Release joins whose requirements are met; propagate fully skipped joins."""
        self.check_timeouts(workflow)
        released = []
        while True:
            batch = self.task_pool.refresh(workflow, self.start_checks)
            released.extend(batch)
            skipped = [
                (task, execution) for task, execution in batch if execution.status == "skipped"
            ]
            for task, execution in skipped:
                self._propagate_skip(workflow, task, execution)
                task.settle_status("all paths resolved")
            if not skipped:
                return released

    # ------------------------------------------------------------------
    # Timeouts
    # ------------------------------------------------------------------

    def check_timeouts(self, workflow: WorkflowDefinition, now: float | None = None) -> None:
        """Call periodically from the host scheduler, also while nodes wait for people."""
        for task in list(self.task_pool.tasks.values()):
            if task.workflow_id != workflow.id:
                continue
            for execution in task.active_executions():
                context = self._bare_context(workflow, task, execution)
                try:
                    check_timeout(context, now)
                except HumanCheckRequired:
                    execution.timing.setdefault("resume_status", execution.status)
                    execution.status = "waiting"
                except TimeoutError as exc:
                    execution.status, execution.error = "failed", str(exc)
                    record_event(task, "node.timeout_failed", {"error": str(exc)}, execution)
            task.settle_status("node timeout or failure")
