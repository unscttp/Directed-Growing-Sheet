"""Portable directed graph data. Scheduling is owned by WorkflowEngine."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from workflow_definitions import EdgeDefinition, NodeDefinition
from workflow_memory_store import atomic_write
from workflow_schema import validate_document


@dataclass
class DirectedGraph:
    nodes: list[NodeDefinition] = field(default_factory=list)
    edges: list[EdgeDefinition] = field(default_factory=list)
    depth: int = 0

    @property
    def node_count(self):
        return len(self.nodes)

    @property
    def edge_count(self):
        return len(self.edges)

    def to_dict(self):
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, raw):
        validate_document(raw, "directed_graph")
        graph = cls(
            [NodeDefinition.from_dict(n) for n in raw["nodes"]],
            [EdgeDefinition.from_dict(e) for e in raw["edges"]],
            raw.get("depth", 0),
        )
        ids = {node.id for node in graph.nodes}
        if len(ids) != len(graph.nodes) or any(
            e.from_node not in ids or e.to_node not in ids for e in graph.edges
        ):
            raise ValueError("Graph has duplicate nodes or an edge referencing an unknown node.")
        return graph

    def save(self, path):
        validated = self.from_dict(self.to_dict())
        atomic_write(
            Path(path),
            json.dumps(validated.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
