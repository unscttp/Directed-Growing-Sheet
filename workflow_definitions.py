from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from workflow_schema import validate_document
from workflow_start_conditions import normalize_start_condition
from workflow_types import JsonDict


@dataclass
class NodeDefinition:
    """
    One node in workflow.json.

    inputs maps local input names to JSON Pointer paths in Task.data.
    outputs maps local output names back to Task.data paths.

    join:
        "any" = one incoming branch is enough.
        "all" = all predecessor branches are required.
    """

    id: str
    type: str
    name: str = ""
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    config: JsonDict = field(default_factory=dict)
    ui: JsonDict = field(default_factory=dict)
    join: str = "any"
    react: JsonDict = field(default_factory=dict)
    allowance: str | None = None
    pre_run: str | None = None
    pre_run_config: JsonDict = field(default_factory=dict)
    post_run: str | None = None
    post_run_config: JsonDict = field(default_factory=dict)
    post_run_human_check: bool = False
    start_condition: JsonDict | None = None
    routing: JsonDict = field(default_factory=dict)
    memory_checkpoints: list[str] = field(default_factory=list)
    timeout_seconds: float | None = None
    timeout_human_check: bool = True
    requires: JsonDict = field(default_factory=dict)
    allow_create_graph: bool = False
    knowledge: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> NodeDefinition:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, raw: JsonDict) -> NodeDefinition:
        validate_document(raw, "node")
        return cls(
            id=raw["id"],
            type=raw["type"],
            name=raw.get("name", ""),
            inputs=dict(raw.get("inputs", {})),
            outputs=dict(raw.get("outputs", {})),
            config=copy.deepcopy(raw.get("config", {})),
            ui=copy.deepcopy(raw.get("ui", {})),
            join=raw.get("join", "any"),
            react=copy.deepcopy(raw.get("react", {})),
            allowance=raw.get("allowance"),
            pre_run=raw.get("pre_run"),
            pre_run_config=copy.deepcopy(raw.get("pre_run_config", {})),
            post_run=raw.get("post_run"),
            post_run_config=copy.deepcopy(raw.get("post_run_config", {})),
            post_run_human_check=raw.get("post_run_human_check", False),
            start_condition=copy.deepcopy(raw.get("start_condition")),
            routing=copy.deepcopy(raw.get("routing", {})),
            memory_checkpoints=copy.deepcopy(raw.get("memory_checkpoints", [])),
            timeout_seconds=raw.get("timeout_seconds"),
            requires=copy.deepcopy(raw.get("requires", {})),
            allow_create_graph=raw.get("allow_create_graph", False),
            knowledge=copy.deepcopy(raw.get("knowledge", {})),
            timeout_human_check=raw.get("timeout_human_check", True),
        )


@dataclass
class EdgeDefinition:
    """
    A directed edge.

    Example condition:
        {"eq": ["/classification/type", "invoice"]}
    """

    from_node: str
    to_node: str
    when: JsonDict | None = None
    otherwise: bool = False
    label: str = ""
    entry_check: str | None = None
    entry_check_config: JsonDict = field(default_factory=dict)
    id: str = ""
    max_traversals: int | None = None

    @classmethod
    def from_dict(cls, raw: JsonDict) -> EdgeDefinition:
        validate_document(raw, "edge")
        return cls(
            from_node=raw["from"],
            to_node=raw["to"],
            when=copy.deepcopy(raw.get("when")),
            otherwise=raw.get("otherwise", False),
            label=raw.get("label", ""),
            entry_check=raw.get("entry_check"),
            entry_check_config=copy.deepcopy(raw.get("entry_check_config", {})),
            id=raw.get("id", ""),
            max_traversals=raw.get("max_traversals"),
        )

    def to_dict(self) -> JsonDict:
        result: JsonDict = {
            "from": self.from_node,
            "to": self.to_node,
        }
        if self.when is not None:
            result["when"] = copy.deepcopy(self.when)
        if self.otherwise:
            result["otherwise"] = True
        if self.label:
            result["label"] = self.label
        if self.entry_check is not None:
            result["entry_check"] = self.entry_check
        if self.entry_check_config:
            result["entry_check_config"] = copy.deepcopy(self.entry_check_config)
        if self.id:
            result["id"] = self.id
        if self.max_traversals is not None:
            result["max_traversals"] = self.max_traversals
        return result


@dataclass
class WorkflowDefinition:
    """
    Portable workflow definition.

    This corresponds closely to workflow.json.
    Runtime implementation details should not be stored here.
    """

    id: str
    name: str
    nodes: list[NodeDefinition]
    edges: list[EdgeDefinition]
    start: list[str]

    format_version: str = "1.0"
    description: str = ""
    inputs: JsonDict = field(default_factory=dict)
    outputs: JsonDict = field(default_factory=dict)
    resources: JsonDict = field(default_factory=dict)
    requirements: JsonDict = field(default_factory=dict)
    layout: JsonDict = field(default_factory=dict)

    def node(self, node_id: str) -> NodeDefinition:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"Unknown node: {node_id}")

    def outgoing(self, node_id: str) -> list[EdgeDefinition]:
        return [edge for edge in self.edges if edge.from_node == node_id]

    def incoming(self, node_id: str) -> list[EdgeDefinition]:
        return [edge for edge in self.edges if edge.to_node == node_id]

    def edge_key(self, edge: EdgeDefinition) -> str:
        if not isinstance(edge.id, str) or (edge.id and not edge.id.strip()):
            raise ValueError("Edge id must be a nonempty string when specified.")
        return edge.id or f"{edge.from_node}->{edge.to_node}"

    def start_rule(self, node_id: str) -> JsonDict | None:
        node = self.node(node_id)
        incoming = [self.edge_key(edge) for edge in self.incoming(node_id)]
        if node.start_condition is not None:
            return normalize_start_condition(node.start_condition, incoming)
        if node.join == "all" and incoming:
            return normalize_start_condition({}, incoming)
        return None

    def validate(self) -> list[str]:
        errors: list[str] = []
        try:
            validate_document(self.to_dict())
        except (ValueError, TypeError) as exc:
            return [str(exc)]
        ids = [node.id for node in self.nodes]
        id_set = set(ids)

        if len(ids) != len(id_set):
            errors.append("Node IDs must be unique.")

        for node in self.nodes:
            defaults = node.routing.get("default_edges", [])
            if not set(defaults) <= {self.edge_key(e) for e in self.outgoing(node.id)}:
                errors.append(
                    f"Node {node.id!r}: routing.default_edges contains unknown outgoing edge IDs."
                )
            if type(node.post_run_human_check) is not bool:
                errors.append(f"Node {node.id!r} post_run_human_check must be a Boolean.")
            try:
                rule = self.start_rule(node.id)
                if rule is not None and node.id in self.start:
                    errors.append(f"Join node {node.id!r} cannot be a workflow start node.")
            except ValueError as exc:
                errors.append(f"Node {node.id!r}: {exc}")
            for phase in ("pre_run", "post_run"):
                reference = getattr(node, phase)
                if reference is not None and (
                    not isinstance(reference, str) or not reference.strip()
                ):
                    errors.append(
                        f"Node {node.id!r} {phase} must be a nonempty function name or null."
                    )
                if not isinstance(getattr(node, f"{phase}_config"), dict):
                    errors.append(f"Node {node.id!r} {phase}_config must be a JSON object.")
            readable = set(node.knowledge.get("read", []))
            if not set(node.knowledge.get("propose", [])) <= readable:
                errors.append(
                    f"Node {node.id!r}: knowledge.propose scopes must also be in knowledge.read."
                )
            if node.allowance is not None and (
                not isinstance(node.allowance, str) or not node.allowance.strip()
            ):
                errors.append(f"Node {node.id!r} allowance must be a nonempty set id or null.")
            if not isinstance(node.react, dict):
                errors.append(f"Node {node.id!r} react must be a JSON object.")
            elif "max_steps" in node.react and (
                type(node.react["max_steps"]) is not int or node.react["max_steps"] < 1
            ):
                errors.append(f"Node {node.id!r} ReAct max_steps must be a positive integer.")
            if node.join not in {"any", "all"}:
                errors.append(f"Node {node.id!r} has invalid join mode {node.join!r}.")

        for node_id in self.start:
            if node_id not in id_set:
                errors.append(f"Start node does not exist: {node_id!r}")

        explicit_edge_ids = [edge.id for edge in self.edges if isinstance(edge.id, str) and edge.id]
        keys = [self.edge_key(edge) for edge in self.edges]
        if len(set(keys)) != len(keys):
            errors.append("All edges must have unique IDs; give parallel edges explicit IDs.")
        if len(set(explicit_edge_ids)) != len(explicit_edge_ids):
            errors.append("Explicit edge IDs must be unique.")
        for edge in self.edges:
            if not isinstance(edge.id, str) or (edge.id and not edge.id.strip()):
                errors.append("Edge id must be a nonempty string when specified.")
            if edge.entry_check is not None and (
                not isinstance(edge.entry_check, str) or not edge.entry_check.strip()
            ):
                errors.append("Edge entry_check must be a nonempty function name or null.")
            if not isinstance(edge.entry_check_config, dict):
                errors.append("Edge entry_check_config must be a JSON object.")
            if edge.from_node not in id_set:
                errors.append(f"Edge source does not exist: {edge.from_node!r}")
            if edge.to_node not in id_set:
                errors.append(f"Edge target does not exist: {edge.to_node!r}")

        return errors

    def assert_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("Invalid workflow:\n- " + "\n- ".join(errors))

    def to_dict(self) -> JsonDict:
        return {
            "format_version": self.format_version,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "inputs": copy.deepcopy(self.inputs),
            "outputs": copy.deepcopy(self.outputs),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "start": list(self.start),
            "resources": copy.deepcopy(self.resources),
            "requirements": copy.deepcopy(self.requirements),
            "layout": copy.deepcopy(self.layout),
        }

    @classmethod
    def from_dict(cls, raw: JsonDict) -> WorkflowDefinition:
        validate_document(raw)
        workflow = cls(
            format_version=raw.get("format_version", "1.0"),
            id=raw["id"],
            name=raw.get("name", raw["id"]),
            description=raw.get("description", ""),
            inputs=copy.deepcopy(raw.get("inputs", {})),
            outputs=copy.deepcopy(raw.get("outputs", {})),
            nodes=[NodeDefinition.from_dict(item) for item in raw.get("nodes", [])],
            edges=[EdgeDefinition.from_dict(item) for item in raw.get("edges", [])],
            start=list(raw.get("start", [])),
            resources=copy.deepcopy(raw.get("resources", {})),
            requirements=copy.deepcopy(raw.get("requirements", {})),
            layout=copy.deepcopy(raw.get("layout", {})),
        )
        workflow.assert_valid()
        return workflow

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> WorkflowDefinition:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
