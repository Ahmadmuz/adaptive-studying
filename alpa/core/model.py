"""The production student model (Phase 1 D1): PFA/IRT hybrid with power-law forgetting.

    State:     per-concept (theta, var, succ, fail, stability, recency)
    Emission:  P(y=1 | S, q) = sigma( a*(theta_eff - b) + beta*min(succ) - delta*min(fail) )
               theta_eff = theta + eta * log R(t)          [forgetting shifts ability]
    Transition: online Bayesian (Laplace/Elo) step on theta, Fisher-information
               shrinkage on var, multiplicative stability update.

All functions are pure: they take a snapshot and return values or new snapshots.
This keeps the math unit-testable and the DB layer a dumb persistence boundary.
"""
from __future__ import annotations

import math
from dataclasses import replace

from .types import ConceptSnapshot, GlobalParams, ItemSpec

_SECONDS_PER_DAY = 86400.0


def sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _days_since(ts: float | None, now: float) -> float:
    if ts is None:
        return 0.0
    return max(0.0, (now - ts) / _SECONDS_PER_DAY)


def retention(snap: ConceptSnapshot, now: float, gp: GlobalParams) -> float:
    """Power-law forgetting curve R(t) = (1 + dt/(9*s))^-alpha.

    Returns 1.0 when the concept has never been successfully practiced:
    nothing learned, nothing to decay.
    """
    if snap.last_success_ts is None:
        return 1.0
    dt = _days_since(snap.last_success_ts, now)
    return (1.0 + dt / (9.0 * snap.stability)) ** (-gp.alpha)


def eff_theta(snap: ConceptSnapshot, now: float, gp: GlobalParams) -> float:
    """Time-decayed effective ability."""
    r = retention(snap, now, gp)
    return snap.theta + gp.eta * math.log(max(r, 1e-9))


def eff_var(snap: ConceptSnapshot, now: float, gp: GlobalParams) -> float:
    """Posterior variance inflated by process noise: uncertainty grows while unseen."""
    return snap.var + gp.rho * _days_since(snap.last_update_ts, now)


def pfa_terms(snap: ConceptSnapshot, gp: GlobalParams) -> tuple[float, float]:
    succ = gp.beta * min(snap.succ, gp.succ_cap)
    fail = -gp.delta * min(snap.fail, gp.fail_cap)
    return succ, fail


def predict(snap: ConceptSnapshot, item: ItemSpec, now: float, gp: GlobalParams) -> dict:
    """Emission model. Returns p and a full feature breakdown for the prediction log."""
    theta_eff = eff_theta(snap, now, gp)
    succ_term, fail_term = pfa_terms(snap, gp)
    z = item.a * (theta_eff - item.b) + succ_term + fail_term
    p = sigmoid(z)
    var_eff = eff_var(snap, now, gp)
    return {
        "p_correct": p,
        "z": z,
        "theta": snap.theta,
        "theta_eff": theta_eff,
        "var_eff": var_eff,
        "retention": retention(snap, now, gp),
        "succ_term": succ_term,
        "fail_term": fail_term,
        "item_a": item.a,
        "item_b": item.b,
    }


def p_correct(snap: ConceptSnapshot, item: ItemSpec, now: float, gp: GlobalParams) -> float:
    return predict(snap, item, now, gp)["p_correct"]


def update(
    snap: ConceptSnapshot,
    item: ItemSpec,
    y: int,
    now: float,
    gp: GlobalParams,
) -> ConceptSnapshot:
    """Transition S_{t+1} = F(S_t, o_t). Online Laplace step:

        precision' = 1/var + a^2 * p * (1 - p)          [Fisher information]
        theta'     = theta + a * (y - p) / precision'   [bounded, monotone]
        var'       = 1 / precision'

    Uses the pre-update p (one Newton step) — exact at 1PL, a tight approximation
    with the PFA terms clamped. Stability grows on success (retrieval strengthens
    memory) and partially resets on failure.
    """
    feats = predict(snap, item, now, gp)
    p = feats["p_correct"]
    prec_prior = 1.0 / max(snap.var, 1e-6)
    info = item.a * item.a * p * (1.0 - p)
    prec = prec_prior + info
    theta_new = snap.theta + item.a * (y - p) / prec
    var_new = 1.0 / prec

    if y == 1:
        stability = min(snap.stability * gp.stability_growth, gp.stability_max)
        last_success = now
    else:
        stability = max(snap.stability * gp.stability_decay, gp.stability_min)
        last_success = snap.last_success_ts

    return replace(
        snap,
        theta=theta_new,
        var=var_new,
        succ=snap.succ + (1 if y == 1 else 0),
        fail=snap.fail + (0 if y == 1 else 1),
        stability=stability,
        last_success_ts=last_success,
        last_update_ts=now,
        exposure=snap.exposure + 1,
    )


def mastery_prob(snap: ConceptSnapshot, now: float, gp: GlobalParams) -> float:
    """P(theta_eff > theta_mastery | history) under the Gaussian posterior."""
    mu = eff_theta(snap, now, gp)
    sd = math.sqrt(max(eff_var(snap, now, gp), 1e-9))
    return normal_cdf((mu - gp.theta_mastery) / sd)


def apply_mastery_hysteresis(snap: ConceptSnapshot, now: float, gp: GlobalParams) -> bool:
    """Mastery as a decision rule with hysteresis (Phase 1 D6)."""
    p = mastery_prob(snap, now, gp)
    if snap.mastered:
        return p >= gp.tau_exit
    return p >= gp.tau_enter


def review_due_days(snap: ConceptSnapshot, now: float, gp: GlobalParams, r_target: float = 0.6) -> float | None:
    """Days from `now` until predicted retention crosses r_target (scheduler input).

    Inverts R(dt) = (1 + dt/(9s))^-alpha. None if never successfully practiced.
    """
    if snap.last_success_ts is None:
        return None
    r_now = retention(snap, now, gp)
    if r_now <= r_target:
        return 0.0
    dt_target_days = 9.0 * snap.stability * (r_target ** (-1.0 / gp.alpha) - 1.0)
    elapsed = _days_since(snap.last_success_ts, now)
    return max(0.0, dt_target_days - elapsed)
