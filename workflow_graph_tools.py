"""The ``graph.create`` tool: reviewed, node-local graphs proposed by a model.

A node with ``allow_create_graph: true`` may propose a small workflow. After the
proposal passes validation and (by default) human review, the same tool call
runs it as a child task and returns the child's data.

Limits on what a generated graph may do:

* every child allowance must be a subset of the creating node's allowance,
  and of every enclosing generated graph's allowance;
* child nodes may only declare MCP servers the creating node declared;
* node types, pre/post-run hooks, edge entry checks and join checks are host
  Python code; a graph may only reference names the host registered with
  ``generated_graphs=True``;
* child nodes may read the creating node's knowledge scopes but never propose
  changes to long-term knowledge;
* human branch selection is not supported inside generated graphs;
* nesting depth and the number of node runs are bounded by global settings.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from workflow_allowances import AllowanceSet
from workflow_approvals import (
    HumanCheckRequired,
    pending_human_checks,
    require_human_check,
    resolve_human_check,
)
from workflow_declarative_tools import digest
from workflow_definitions import WorkflowDefinition
from workflow_executions import Task
from workflow_graphs import DirectedGraph
from workflow_joins import TaskPool
from workflow_memory import append_active_memory
from workflow_memory_store import atomic_write
from workflow_schema import schema_document, validate_document
from workflow_skill_graph_meta import graph_environment, record_run
from workflow_skills import load_skill_graph, open_skill_calls, read_manifest
from workflow_tool_safety import ToolSafetyPolicy
from workflow_tools import ToolDefinition
from workflow_utils import new_id

MAX_PROPOSAL_BYTES = 256 * 1024

TOOL_DESCRIPTION = (
    "Create and immediately execute a node-local directed graph after human review. "
    "Supply nodes, edges, start and a concise explanation of its purpose and effects. "
    "Node types, hooks and checks must be ones the host allows in generated graphs. "
    "For a skill provide skill_invoke_id. Optional save_to_skill asks, after the graph "
    "completed, for a separate review to persist it in that skill package; usage_scenario "
    "must describe when to use it in Markdown. Runtime controls nesting depth."
)


def _host_references(graph: DirectedGraph) -> list[tuple[str, str]]:
    """Every (kind, name) of host Python code that the graph would run."""
    references = []
    for node in graph.nodes:
        references.append(("handler", node.type))
        if node.pre_run is not None:
            references.append(("pre_run", node.pre_run))
        if node.post_run is not None:
            references.append(("post_run", node.post_run))
        check = (node.start_condition or {}).get("check")
        if check is not None:
            references.append(("start_check", check))
    for edge in graph.edges:
        if edge.entry_check is not None:
            references.append(("entry_check", edge.entry_check))
    return references


class GraphTools:
    def __init__(self, engine) -> None:
        self.engine = engine

    def get_tool(self, name: str) -> ToolDefinition:
        if name != "graph.create":
            raise KeyError(name)
        definitions = schema_document()["$defs"]
        schema = copy.deepcopy(definitions["graph_request"])
        # Expand node and edge references so the model sees a self-contained schema.
        schema["properties"]["nodes"]["items"] = copy.deepcopy(definitions["node"])
        schema["properties"]["edges"]["items"] = copy.deepcopy(definitions["edge"])
        schema["$defs"] = copy.deepcopy(definitions)
        return ToolDefinition(name, "high", description=TOOL_DESCRIPTION, input_schema=schema)

    # -- validation --------------------------------------------------------

    def _check_host_references(self, graph: DirectedGraph) -> None:
        allowed = self.engine.generated_graph_names
        denied = sorted(
            f"{kind} {name!r}"
            for kind, name in set(_host_references(graph))
            if name not in allowed[kind]
        )
        if denied:
            raise PermissionError(
                "Generated graphs may only use host code registered with "
                "generated_graphs=True; not allowed: " + ", ".join(denied)
            )
        for kind, name in _host_references(graph):
            if kind == "handler" and name not in self.engine.handlers:
                raise ValueError(f"Unknown child node handler: {name}")

    def _check_allowances(self, context, graph: DirectedGraph) -> None:
        parent_policy = self.engine.allowances.resolve(context.node.allowance)
        parent_servers = set(context.node.requires.get("mcp_servers", []))
        for node in graph.nodes:
            if node.allowance is None:
                node.allowance = context.node.allowance
            policy = self.engine.allowances.resolve(node.allowance)
            if parent_policy is not None and policy is None:
                raise PermissionError("Child graph allowance exceeds parent actions.")
            if policy is not None:
                policy.assert_subset_of(parent_policy)
            for _, ancestor in self.engine.graph_ceiling_policies():
                if policy is None:
                    raise PermissionError("Child graph allowance exceeds ancestor.")
                policy.assert_subset_of(ancestor)
            if not set(node.requires.get("mcp_servers", [])) <= parent_servers:
                raise PermissionError("Child graph MCP dependencies exceed parent.")
            if node.routing.get("mode") == "human":
                raise ValueError("Generated graphs support default or LLM branch selection only.")
            self._check_knowledge(context, node)

    @staticmethod
    def _check_knowledge(context, node) -> None:
        """Generated graphs may read the creating node's knowledge, never propose."""
        parent_read = context.node.knowledge.get("read", [])
        if not node.knowledge:
            node.knowledge = {"read": list(parent_read)} if parent_read else {}
        elif node.knowledge.get("propose"):
            raise PermissionError("Generated graphs cannot propose knowledge changes.")
        elif not set(node.knowledge.get("read", [])) <= set(parent_read):
            raise PermissionError("Child graph knowledge scopes exceed parent.")

    def _skill_for(self, context, args) -> dict | None:
        calls = open_skill_calls(context)
        invoke = args.get("skill_invoke_id")
        if invoke is None and len(calls) == 1:
            invoke = next(iter(calls))
        if invoke is None and len(calls) > 1:
            raise ValueError("Choose skill_invoke_id when multiple skills are open.")
        if invoke is None:
            return None
        if invoke not in calls:
            raise ValueError("Skill invocation is closed, foreign or unknown.")
        return {"invoke_id": invoke, "name": calls[invoke]["skill"]}

    def _package_root(self, skill: dict) -> Path:
        packages = getattr(self.engine.tools, "skill_packages", {})
        package = packages.get(skill["name"])
        if package is None:
            raise ValueError("Skill package is not registered with the current tool module.")
        root = Path(package["root"])
        _, _, current_hash = read_manifest(root)
        if current_hash != package["manifest_hash"]:
            raise PermissionError("Skill manifest changed since registration.")
        return root

    def _proposal(self, context, args, settings, *, skill_graph=None):
        size = len(json.dumps(args, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        if size > MAX_PROPOSAL_BYTES:
            raise ValueError("Graph proposal exceeds 256 KiB.")
        validate_document(args, "graph_request")
        if not args["explanation"].strip():
            raise ValueError("Graph creation requires an explanation.")
        # A graph saved in a skill is permitted by the skill's run_graph action instead.
        if skill_graph is None and not context.node.allow_create_graph:
            raise PermissionError("This node does not allow graph creation.")
        parent_depth = context.task.metadata.get("_graph_depth", 0)
        if type(parent_depth) is not int or parent_depth < 0:
            raise ValueError("Invalid runtime graph depth.")
        if parent_depth >= settings["graph_max_depth"]:
            raise PermissionError("Graph nesting depth limit reached.")
        graph = DirectedGraph.from_dict(
            {"nodes": args["nodes"], "edges": args["edges"], "depth": parent_depth + 1}
        )
        self._check_host_references(graph)
        self._check_allowances(context, graph)
        WorkflowDefinition(
            "proposal", "Generated graph", graph.nodes, graph.edges, args["start"]
        ).assert_valid()
        skill = self._skill_for(context, args)
        if args.get("save_to_skill"):
            if skill is None:
                raise ValueError("Saving a graph requires an active skill invocation.")
            if not args.get("usage_scenario", "").strip():
                raise ValueError(
                    "Saving a skill graph requires a nonempty usage_scenario in Markdown."
                )
        return graph, skill

    # -- execution ---------------------------------------------------------

    def execute(self, context, tool, args):
        return self._execute(context, args)

    def run_skill_graph(self, context, invoke_id: str, graph_id: str):
        """Run ``graphs/<graph_id>.json`` of the skill opened as ``invoke_id``.

        Called by ``SkillGraphInterface``. The saved graph goes through the same
        checks and review as a ``graph.create`` proposal, except that the node
        needs no ``allow_create_graph``.
        """
        skill = self._skill_for(context, {"skill_invoke_id": invoke_id})
        root = self._package_root(skill)
        saved = load_skill_graph(root, graph_id)
        args = {
            "nodes": saved["graph"]["nodes"],
            "edges": saved["graph"]["edges"],
            "start": saved["start"],
            "explanation": saved["explanation"],
            "skill_invoke_id": invoke_id,
        }
        key = digest({**args, "skill_graph": graph_id})
        # Each run is recorded once in the graph's metadata, also across a resume.
        recorded = context.execution.memory_state.setdefault("skill_graph_runs", {})
        try:
            result = self._execute(context, args, skill_graph=graph_id)
        except HumanCheckRequired:
            raise
        except Exception as exc:
            if key not in recorded:
                recorded[key] = "failed"
                self._record_run(context, root, graph_id, saved, key, "failed", str(exc))
            raise
        if key not in recorded:
            recorded[key] = "completed"
            self._record_run(context, root, graph_id, saved, key, "completed")
        return result

    def _record_run(self, context, root, graph_id, saved, key, outcome, error=None) -> None:
        record = context.execution.memory_state.get("graphs", {}).get(key) or {}
        try:
            record_run(
                root,
                graph_id,
                graph_environment(self.engine, saved["graph"]),
                outcome,
                error=error,
                steps=record.get("steps"),
                child_task_id=(record.get("task") or {}).get("id"),
            )
        except (OSError, PermissionError) as exc:
            context.record_activity(
                "skill.graph.meta_failed", {"graph_id": graph_id, "error": str(exc)}
            )

    def _execute(self, context, args, skill_graph=None):
        settings = self.engine.settings()
        graph, skill = self._proposal(context, args, settings, skill_graph=skill_graph)
        records = context.execution.memory_state.setdefault("graphs", {})
        key = digest({**args, "skill_graph": skill_graph} if skill_graph else args)
        record = records.get(key)
        if record is None:
            record = {
                "id": new_id("graph"),
                "graph": graph.to_dict(),
                "skill": skill,
                "start": copy.deepcopy(args["start"]),
                "explanation": args["explanation"],
                "usage_scenario": args.get("usage_scenario"),
                "skill_graph": skill_graph,
                "parent_allowance": context.allowance_snapshot(),
                "approved": False,
                "steps": 0,
            }
            records[key] = record
            record["request_event"] = context.record_activity(
                "graph.requested", copy.deepcopy(record)
            )
        if record["graph"] != graph.to_dict() or record["skill"] != skill:
            raise PermissionError("Graph context changed; submit a new proposal for review.")
        if not record["approved"]:
            self._approve(context, record, settings)
        if record.get("error"):
            raise RuntimeError(record["error"])
        if record.get("result") is None:
            self._run(context, record, settings)
            context.record_activity(
                "graph.completed",
                {"graph_id": record["id"], "child_task_id": record["task"]["id"]},
            )
        # Saved only once the graph has run successfully: a saved graph records a run
        # that worked. Declining the save keeps the result.
        if args.get("save_to_skill") and "save_outcome" not in record:
            self._save_skill(context, record, settings)
            record["result"]["saved_path"] = record.get("saved")
        return copy.deepcopy(record["result"])

    def _approve(self, context, record, settings) -> None:
        if settings["graph_creation_human_check"]:
            require_human_check(
                context,
                "graph.create:" + record["id"],
                {
                    "kind": "graph_creation",
                    "graph": record["graph"],
                    "start": record["start"],
                    "explanation": record["explanation"],
                    "usage_scenario": record.get("usage_scenario"),
                    "skill": record["skill"],
                    "skill_graph": record.get("skill_graph"),
                },
            )
        record["approved"] = True
        event = context.record_activity(
            "graph.created",
            {
                "graph_id": record["id"],
                "graph": record["graph"],
                "skill": record["skill"],
                "explanation": record["explanation"],
            },
        )
        append_active_memory(
            context,
            record["explanation"],
            checkpoint="graph_creation",
            kind="decision",
            references=[record["request_event"], event],
        )

    def _safety_policy(self) -> ToolSafetyPolicy:
        """The host's configured file safety policy, or the bundled default."""
        policy = getattr(self.engine.tools, "safety_policy", None)
        return policy if isinstance(policy, ToolSafetyPolicy) else ToolSafetyPolicy()

    def _save_skill(self, context, record, settings) -> None:
        root = self._package_root(record["skill"])
        _, _, current_hash = read_manifest(root)
        directory = root / "graphs"
        target = directory / (record["id"] + ".json")
        scenario = target.with_suffix(".md")
        if (
            directory.is_symlink()
            or (hasattr(directory, "is_junction") and directory.is_junction())
            or target.is_symlink()
            or scenario.is_symlink()
        ):
            raise PermissionError("Skill graph destination cannot be a link.")
        if not target.resolve().is_relative_to(root.resolve()):
            raise PermissionError("Skill graph destination escapes package.")
        safety = self._safety_policy()
        writer = ToolDefinition("fs.write_text", "medium")
        for destination in (target, scenario):
            safety.check_system_path(context, writer, str(destination), root, None)
        if settings["graph_save_human_check"]:
            try:
                self._review_save(context, record, target, scenario, current_hash)
            except PermissionError:
                record["save_outcome"] = "declined"
                context.record_activity(
                    "skill.graph.save_declined", {"graph_id": record["id"], **record["skill"]}
                )
                return
        self._write_skill_graph(context, record, target, scenario)

    @staticmethod
    def _review_save(context, record, target, scenario, current_hash) -> None:
        require_human_check(
            context,
            "graph.save:" + record["id"],
            {
                "kind": "skill_graph_save",
                "on_reject": "continue",
                "path": str(target),
                "graph": record["graph"],
                "start": record["start"],
                "explanation": record["explanation"],
                "usage_scenario": record.get("usage_scenario"),
                "scenario_path": str(scenario),
                "skill": record["skill"],
                "manifest_sha256": current_hash,
            },
        )

    def _write_skill_graph(self, context, record, target, scenario) -> None:
        directory = target.parent
        if (
            not isinstance(record.get("usage_scenario"), str)
            or not record["usage_scenario"].strip()
        ):
            raise ValueError("Skill graph needs a usage scenario.")
        markdown = record["usage_scenario"].encode("utf-8")
        data = json.dumps(
            {
                "usage_scenario_file": scenario.name,
                "usage_scenario_sha256": hashlib.sha256(markdown).hexdigest(),
                "graph": record["graph"],
                "start": record["start"],
                "explanation": record["explanation"],
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError("Skill graph destination already contains different data.")
        if scenario.exists() and scenario.read_bytes() != markdown:
            raise FileExistsError("Skill graph scenario already contains different data.")
        directory.mkdir(exist_ok=True)
        # Publish the JSON last, so a visible graph always has its scenario file.
        atomic_write(scenario, markdown)
        atomic_write(target, data)
        record["saved"] = str(target)
        record["save_outcome"] = "saved"
        # The run that produced the graph is its first recorded run.
        try:
            record_run(
                target.parent.parent,
                record["id"],
                graph_environment(self.engine, record["graph"]),
                "completed",
                steps=record.get("steps"),
                child_task_id=(record.get("task") or {}).get("id"),
                saved=True,
            )
        except (OSError, PermissionError) as exc:
            context.record_activity(
                "skill.graph.meta_failed", {"graph_id": record["id"], "error": str(exc)}
            )
        context.record_activity(
            "skill.graph.saved",
            {
                "graph_id": record["id"],
                "path": str(target),
                "scenario_path": str(scenario),
                **record["skill"],
            },
        )

    def _bridge_reviews(self, context, child: Task, graph_id: str) -> None:
        """Ask the parent's reviewer for each child approval, then pass the decision down."""
        for request in pending_human_checks(child):
            key = "graph.child:" + request["id"]
            require_human_check(
                context,
                key,
                {
                    "kind": "child_graph_review",
                    "graph_id": graph_id,
                    "child_task_id": child.id,
                    "child_request": request,
                },
            )
            accepted = next(
                r
                for r in reversed(context.execution.human_checks)
                if r["key"] == key and r["status"] == "consumed"
            )
            resolve_human_check(
                child,
                request["id"],
                True,
                reviewer=accepted["reviewer"],
                comment=accepted["comment"],
            )

    def _child_engine(self, context, record):
        child_engine = copy.copy(self.engine)
        # A separate queue keeps the host scheduler from running child nodes on its own.
        child_engine.task_pool = TaskPool()
        allowance_snapshot = record.get("parent_allowance", context.allowance_snapshot())
        child_engine.graph_permission_ceilings = self.engine.graph_permission_ceilings + [
            (context, allowance_snapshot)
        ]
        child_engine.graph_tools = GraphTools(child_engine)
        return child_engine, allowance_snapshot

    def _recheck_allowances(self, context, graph, allowance_snapshot) -> None:
        """Parent allowances may have changed while the child was paused."""
        parents = [self.engine.allowances.resolve(context.node.allowance)]
        parents.extend(policy for _, policy in self.engine.graph_ceiling_policies())
        if allowance_snapshot is not None:
            parents.append(AllowanceSet.from_dict(allowance_snapshot))
        for node in graph.nodes:
            policy = self.engine.allowances.resolve(node.allowance)
            for parent in parents:
                if parent is not None and policy is None:
                    raise PermissionError("Child graph allowance exceeds parent.")
                if policy is not None:
                    policy.assert_subset_of(parent)

    def _run(self, context, record, settings):
        graph = DirectedGraph.from_dict(record["graph"])
        workflow = WorkflowDefinition(
            record["id"], "Generated graph", graph.nodes, graph.edges, record["start"]
        )
        if "task" in record:
            child = Task.from_dict(record["task"])
        else:
            child = Task.create(workflow, data=context.task.data)
            child.metadata.update(
                _graph_depth=graph.depth,
                parent_task_id=context.task.id,
                parent_execution_id=context.execution.id,
            )
        child_engine, allowance_snapshot = self._child_engine(context, record)
        child_engine.task_pool.add(child)
        try:
            self._bridge_reviews(context, child, record["id"])
            while child.active_executions():
                context.check_deadline()
                child_engine.refresh_start_conditions(workflow)
                ready = [e for e in child.executions if e.status == "ready"]
                if not ready:
                    self._bridge_reviews(context, child, record["id"])
                    ready = [e for e in child.executions if e.status == "ready"]
                    if not ready:
                        raise RuntimeError(
                            "Child graph is waiting without a resolvable human approval."
                        )
                if record["steps"] >= settings["graph_max_steps"]:
                    raise RuntimeError("Child graph execution step budget exhausted.")
                self._recheck_allowances(context, graph, allowance_snapshot)
                record["steps"] += 1
                child_engine.run_execution(workflow, child, ready[0])
                self._bridge_reviews(context, child, record["id"])
            if child.status != "completed":
                errors = "; ".join(e.error or "" for e in child.executions if e.status == "failed")
                raise RuntimeError("Child graph failed: " + errors)
            result = {
                "graph_id": record["id"],
                "status": "completed",
                "depth": graph.depth,
                "result": copy.deepcopy(child.data),
                "saved_path": record.get("saved"),
            }
            record["result"] = result
            return copy.deepcopy(result)
        except HumanCheckRequired:
            raise
        except Exception as exc:
            record["error"] = str(exc)
            raise
        finally:
            record["task"] = child.to_dict()
            self._merge_child_memory(context, child)
            if record.get("error"):
                context.record_activity(
                    "graph.failed", {"graph_id": record["id"], "error": record["error"]}
                )

    @staticmethod
    def _merge_child_memory(context, child: Task) -> None:
        parent = context.task
        seen = {event["id"] for event in parent.events}
        parent.events.extend(copy.deepcopy(e) for e in child.events if e["id"] not in seen)
        archived = {note["id"] for note in child.active_memory["archive"]}
        parent.active_memory["current"] = [
            note for note in parent.active_memory["current"] if note["id"] not in archived
        ]
        known = {note["id"] for section in parent.active_memory.values() for note in section}
        for section in ("current", "archive"):
            parent.active_memory[section].extend(
                copy.deepcopy(note)
                for note in child.active_memory[section]
                if note["id"] not in known
            )
        parent.revision += 1
