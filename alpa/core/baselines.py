"""Interpretable baselines kept for Phase 4 comparison: BKT and raw PFA.

Fitting (EM for BKT, logistic regression for PFA) runs offline on logged data when
enough attempts exist; here they ship with default parameters so their interfaces
are frozen and the comparison harness can be built against stable code.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import sigmoid


@dataclass
class BKTParams:
    p_l0: float = 0.10   # P(learned) before first opportunity
    p_t: float = 0.10    # P(learn) per opportunity while unlearned
    p_g: float = 0.20    # P(guess) when unlearned
    p_s: float = 0.10    # P(slip) when learned


@dataclass
class BKTState:
    p_learned: float


def bkt_update(state: BKTState, y: int, params: BKTParams) -> BKTState:
    """Classic HMM posterior update + learning transition (Corbett & Anderson 1994)."""
    l = state.p_learned
    g, s = params.p_g, params.p_s
    if y == 1:
        num = l * (1.0 - s)
        denom = l * (1.0 - s) + (1.0 - l) * g
    else:
        num = l * s
        denom = l * s + (1.0 - l) * (1.0 - g)
    post = num / denom if denom > 0 else l
    return BKTState(p_learned=post + (1.0 - post) * params.p_t)


def bkt_predict(state: BKTState, params: BKTParams) -> float:
    l = state.p_learned
    return l * (1.0 - params.p_s) + (1.0 - l) * params.p_g


def pfa_predict(succ: int, fail: int, gamma: float = 0.0, beta: float = 0.25, delta: float = 0.35) -> float:
    """Pavlik & Anderson (2005), item intercept folded into gamma."""
    return sigmoid(gamma + beta * succ - delta * fail)
