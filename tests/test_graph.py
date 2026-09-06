"""Concept graph: traversal, provenance gating, cycles, serialization."""
import pytest

from alpa.core.graph import (ConceptGraph, ConceptNode, Edge, PREREQUISITE,
                             RELATED, PART_OF)


def diamond() -> ConceptGraph:
    g = ConceptGraph(graph_version=1)
    for i, code in enumerate(["a", "b", "c", "d"], start=1):
        g.add_node(ConceptNode(id=i, code=code, title=code))
    g.add_edge(Edge(1, 2))
    g.add_edge(Edge(1, 3))
    g.add_edge(Edge(2, 4))
    g.add_edge(Edge(3, 4))
    return g


def test_topo_order_valid():
    order = diamond().topo_order()
    assert order[0] == 1 and order[-1] == 4
    assert order.index(2) > order.index(1) and order.index(3) > order.index(1)


def test_cycle_detection():
    g = diamond()
    g.add_edge(Edge(4, 1))
    with pytest.raises(ValueError):
        g.topo_order()


def test_frontier_follows_mastery():
    g = diamond()
    assert g.frontier(set()) == [1]
    assert g.frontier({1}) == [2, 3]
    assert g.frontier({1, 2, 3}) == [4]
    assert g.frontier({1, 2, 3, 4}) == []


def test_llm_edges_do_not_steer_adaptation_until_reviewed():
    g = diamond()
    g.add_edge(Edge(1, 4, kind=PREREQUISITE, confidence=0.9,
                    provenance="llm:gpt-x", human_reviewed=False))
    # default traversal (what the engine uses): unreviewed LLM edge excluded
    assert g.prereqs(4) == [2, 3]
    # explicit opt-in includes it
    assert g.prereqs(4, include_unreviewed_llm=True) == [2, 3, 1]
    # after human review it is traversable by default
    g.edges[-1].human_reviewed = True
    assert g.prereqs(4) == [2, 3, 1]


def test_low_confidence_filtered():
    g = diamond()
    g.add_edge(Edge(1, 4, confidence=0.2, provenance="llm:m", human_reviewed=True))
    assert g.prereqs(4, min_confidence=0.5) == [2, 3]


def test_edge_kinds_are_typed():
    g = diamond()
    g.add_edge(Edge(1, 4, kind=RELATED))
    g.add_edge(Edge(2, 4, kind=PART_OF))
    with pytest.raises(ValueError):
        g.add_edge(Edge(1, 2, kind="SUPPORTS"))
    # traversal defaults to PREREQUISITE only
    assert len(g.traversable()) == 4
    assert len(g.traversable(kinds=(PREREQUISITE, RELATED, PART_OF))) == 6


def test_serialization_roundtrip():
    g = diamond()
    g.add_edge(Edge(1, 4, kind=RELATED, provenance="llm:m", human_reviewed=False))
    g2 = ConceptGraph.from_json(g.to_json())
    assert g2.to_json() == g.to_json()
    assert g2.frontier({1}) == g.frontier({1})


def test_missing_endpoint_rejected():
    g = diamond()
    with pytest.raises(ValueError):
        g.add_edge(Edge(1, 99))
