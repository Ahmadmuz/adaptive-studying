"""Cold-start diagnostic: maximum-Fisher-information item selection with a hard budget.

For the 2PL emission, Fisher information I(theta) = a^2 * p * (1-p) is maximized at
b ~= theta with high discrimination. Selecting items this way shrinks posterior
variance fastest per item, which is exactly what lets us cap the diagnostic at a
small budget instead of running long fixed batteries (Phase 1 D-cold).
"""
from __future__ import annotations

from . import model
from .types import ConceptSnapshot, GlobalParams, ItemSpec


def fisher_information(a: float, p: float) -> float:
    return a * a * p * (1.0 - p)


def pick_diagnostic_item(
    snapshots: dict[int, ConceptSnapshot],
    items: list[ItemSpec],
    attempted_item_ids: set[int],
    fresh_factory,
    now: float,
    gp: GlobalParams,
) -> tuple[ItemSpec, float] | None:
    """Pick the unattempted item with maximum expected information.

    `fresh_factory(concept_id) -> ConceptSnapshot` supplies the population-prior
    snapshot for concepts the student has no state for yet.
    Prefers items of `kind == "diagnostic"` when any candidate exists.
    """
    best: tuple[float, ItemSpec, float] | None = None  # (info, item, p)
    best_diag: tuple[float, ItemSpec, float] | None = None
    for item in items:
        if item.item_id in attempted_item_ids:
            continue
        snap = snapshots.get(item.concept_ids[0]) or fresh_factory(item.concept_ids[0])
        p = model.p_correct(snap, item, now, gp)
        info = fisher_information(item.a, p)
        cand = (info, item, p)
        if item.kind == "diagnostic":
            if best_diag is None or info > best_diag[0]:
                best_diag = cand
        if best is None or info > best[0]:
            best = cand
    chosen = best_diag or best
    if chosen is None:
        return None
    return chosen[1], chosen[2]


def diagnostic_budget(total_attempts: int, budget: int = 8) -> bool:
    """MVP stopping rule: fixed budget. (Variance-threshold stopping is a strict
    improvement and is deferred until real variance trajectories are observed.)"""
    return total_attempts < budget
