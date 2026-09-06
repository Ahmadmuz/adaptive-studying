"""Concept graph abstraction (Milestone 4).

Typed, provenance-aware edges and the traversal primitives the adaptive engine
needs. Policy: the adaptive engine traverses edges that are human-reviewed, or
explicitly opted-in LLM edges above a confidence floor — LLM-proposed edges
never silently steer adaptation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

PREREQUISITE = "PREREQUISITE"
RELATED = "RELATED"
PART_OF = "PART_OF"
DEPENDS_ON = "DEPENDS_ON"
SPECIAL_CASE_OF = "SPECIAL_CASE_OF"
EDGE_KINDS = {PREREQUISITE, RELATED, PART_OF, DEPENDS_ON, SPECIAL_CASE_OF}


@dataclass
class ConceptNode:
    id: int
    code: str
    title: str
    description: str = ""
    difficulty: float = 0.0        # population difficulty prior (IRT-b scale)
    metadata: dict = field(default_factory=dict)


@dataclass
class Edge:
    src: int                       # prerequisite / source
    dst: int                       # dependent concept
    kind: str = PREREQUISITE
    confidence: float = 1.0
    provenance: str = "human"      # "human" | "llm:<provider/model>"
    human_reviewed: bool = True
    id: int | None = None          # DB row id when loaded from persistence

    @property
    def is_llm(self) -> bool:
        return self.provenance.startswith("llm:")


class ConceptGraph:
    def __init__(self, graph_version: int = 1):
        self.graph_version = graph_version
        self.nodes: dict[int, ConceptNode] = {}
        self.edges: list[Edge] = []

    # ------------------------------------------------------------- building
    def add_node(self, node: ConceptNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.kind not in EDGE_KINDS:
            raise ValueError(f"unknown edge kind {edge.kind}")
        if edge.src not in self.nodes or edge.dst not in self.nodes:
            raise ValueError("edge endpoints must exist in the graph")
        self.edges.append(edge)

    # ------------------------------------------------------------- querying
    def traversable(self, kinds: Iterable[str] = (PREREQUISITE,),
                    min_confidence: float = 0.0,
                    include_unreviewed_llm: bool = False) -> list[Edge]:
        """Edges the adaptive engine may act on."""
        out = []
        for e in self.edges:
            if e.kind not in kinds or e.confidence < min_confidence:
                continue
            if e.is_llm and not e.human_reviewed and not include_unreviewed_llm:
                continue
            out.append(e)
        return out

    def prereqs(self, concept_id: int, kinds: Iterable[str] = (PREREQUISITE,),
                **filters) -> list[int]:
        return [e.src for e in self.traversable(kinds, **filters)
                if e.dst == concept_id]

    def dependents(self, concept_id: int, kinds: Iterable[str] = (PREREQUISITE,),
                   **filters) -> list[int]:
        return [e.dst for e in self.traversable(kinds, **filters)
                if e.src == concept_id]

    def topo_order(self) -> list[int]:
        """Kahn's algorithm over traversable prerequisite edges; deterministic
        tie-break by node id."""
        edges = [(e.src, e.dst) for e in self.traversable((PREREQUISITE,))]
        indeg = {n: 0 for n in self.nodes}
        children: dict[int, list[int]] = {n: [] for n in self.nodes}
        for s, d in edges:
            indeg[d] += 1
            children[s].append(d)
        ready = sorted(n for n, d in indeg.items() if d == 0)
        order = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for c in sorted(children[n]):
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
            ready.sort()
        if len(order) != len(self.nodes):
            raise ValueError("prerequisite graph has a cycle")
        return order

    def frontier(self, mastered: set[int]) -> list[int]:
        """Unmastered concepts whose traversable prerequisites are all mastered,
        in topological order — the adaptive engine's candidate set."""
        return [c for c in self.topo_order()
                if c not in mastered and set(self.prereqs(c)) <= mastered]

    # -------------------------------------------------------- serialization
    def to_json(self) -> str:
        return json.dumps({
            "graph_version": self.graph_version,
            "nodes": [n.__dict__ for n in self.nodes.values()],
            "edges": [e.__dict__ for e in self.edges],
        }, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "ConceptGraph":
        d = json.loads(raw)
        g = cls(graph_version=d["graph_version"])
        for n in d["nodes"]:
            g.add_node(ConceptNode(**n))
        for e in d["edges"]:
            g.add_edge(Edge(**e))
        return g
