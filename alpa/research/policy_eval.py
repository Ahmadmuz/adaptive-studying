"""Synthetic-student policy evaluation (Milestone 10 / Phase M).

Simulates configurable learner profiles with PLANTED dynamics against the real
policy machinery (alpa.core.policy + alpa.core.model + mechanics item bank),
so we can check whether the adaptive system behaves as intended:

    mastery gain, uncertainty reduction, success rate, coverage,
    difficulty progression, intervention distribution, prerequisite recovery.

HONESTY CONTRACT: every number produced here is evidence about the ALGORITHM
under known ground truth. It is never evidence that humans learn better.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core import model, policy
from ..core.state import Observation, StudentState
from ..core.types import (AttemptRecord, ConceptSnapshot, GlobalParams, ItemSpec,
                          SIMPLIFY, BREAK_TASK, REVISIT_PREREQUISITE)
from ..seed.mechanics import CONCEPTS, ITEMS

CODES = [c[0] for c in CONCEPTS]
CID = {c: i for i, c in enumerate(CODES)}
PREREQS = {CID[c[0]]: [CID[p] for p in c[3]] for c in CONCEPTS}
TOPO = list(range(len(CODES)))


def item_bank() -> dict[int, list[ItemSpec]]:
    bank: dict[int, list[ItemSpec]] = {}
    for i, (code, _kind, stem, choices, correct, b, a) in enumerate(ITEMS):
        bank.setdefault(CID[code], []).append(ItemSpec(
            item_id=i, stem=stem, choices=tuple(choices), correct_index=correct,
            concept_ids=(CID[code],), b=b, a=a, kind=_kind))
    return bank


PROFILES = {
    "fast":          dict(learn=0.55, prior=0.0, forget=0.0, noise=0.6),
    "slow":          dict(learn=0.15, prior=0.0, forget=0.0, noise=0.6),
    "high_prior":    dict(learn=0.35, prior=1.2, forget=0.0, noise=0.6),
    "low_prior":     dict(learn=0.35, prior=-1.2, forget=0.0, noise=0.6),
    "forgetting":    dict(learn=0.40, prior=0.0, forget=0.35, noise=0.6),
    "inconsistent":  dict(learn=0.35, prior=0.0, forget=0.0, noise=1.8),
    "weak_concepts": dict(learn=0.35, prior=0.0, forget=0.0, noise=0.6),
}


@dataclass
class SimStudent:
    profile: str
    theta: np.ndarray
    last_practice: np.ndarray
    gp: GlobalParams = field(default_factory=GlobalParams)

    @classmethod
    def create(cls, profile: str, rng: np.random.Generator) -> "SimStudent":
        pr = PROFILES[profile]
        theta = np.full(len(CODES), pr["prior"] + rng.normal(0, 0.25))
        if profile == "weak_concepts":
            theta[list(rng.choice(len(CODES), size=3, replace=False))] -= 1.0
        return cls(profile=profile, theta=theta,
                   last_practice=np.full(len(CODES), -50.0))

    def p_correct(self, concept_id: int, b: float, step: float) -> float:
        pr = PROFILES[self.profile]
        gap = step - self.last_practice[concept_id]
        z = self.theta[concept_id] - pr["forget"] * math.log1p(max(gap, 0)) - b
        return 1.0 / (1.0 + math.exp(-np.clip(z, -30, 30)))

    def respond(self, concept_id: int, b: float, step: float,
                rng: np.random.Generator) -> bool:
        y = rng.random() < self.p_correct(concept_id, b, step)
        pr = PROFILES[self.profile]
        self.theta[concept_id] += pr["learn"] if y else -0.3 * pr["learn"]
        self.last_practice[concept_id] = step
        return y

    def teach(self, concept_id: int, n_times: int) -> None:
        """Planted instructional effect: teaching raises ability at half the
        practice rate, with geometrically diminishing returns per repeat.
        EXPLICIT ASSUMPTION of the simulator — documented, not hidden."""
        pr = PROFILES[self.profile]
        gain = 0.5 * pr["learn"] * (0.5 ** n_times)
        self.theta[concept_id] += gain


# ---------------------------------------------------------------- policies ---

def policy_random(state, ctx_common, rng):
    cid = int(rng.integers(0, len(CODES)))
    items = ctx_common["items_by_concept"].get(cid, [])
    if not items:
        return None
    it = items[int(rng.integers(0, len(items)))]
    return ("Q_MED", cid, it)


def policy_fixed(state, ctx_common, rng):
    """Non-adaptive baseline: medium-band item on the first unmastered concept;
    no teaching, no reviews, no error handling."""
    gp = GlobalParams()
    mastered = {c for c, s in state.concepts.items() if s.mastered}
    frontier = [c for c in TOPO if c not in mastered
                and set(PREREQS[c]) <= mastered]
    if not frontier:
        return None
    cid = frontier[0]
    snap = state.concepts[cid]
    items = [it for it in ctx_common["items_by_concept"].get(cid, [])
             if it.item_id not in ctx_common["attempted"]] \
        or ctx_common["items_by_concept"].get(cid, [])
    if not items:
        return None
    best = min(items, key=lambda it: abs(model.p_correct(snap, it, ctx_common["now"], gp) - 0.55))
    return ("Q_MED", cid, best)


REMEDIATION_ACTIONS = (SIMPLIFY, BREAK_TASK, REVISIT_PREREQUISITE)


def policy_rules(state, ctx_common, rng, disable_info_gain: bool = False):
    """The production rule policy operating on the live StudentState."""
    if disable_info_gain:
        orig = policy.INFO_GAIN_VAR_MIN
        policy.INFO_GAIN_VAR_MIN = 1e9
    try:
        ctx = policy.PolicyContext(
            snapshots=dict(state.concepts), topo_order=TOPO, prereqs=PREREQS,
            items_by_concept=ctx_common["items_by_concept"],
            attempted_item_ids=ctx_common["attempted"],
            recent_attempts=ctx_common["recent"],
            content_shown=ctx_common["content_shown"],
            remediated=ctx_common["remediated"],
            now=ctx_common["now"], session_attempts=ctx_common["session_attempts"],
            gp=GlobalParams(), rng=rng, epsilon=0.0)
        return policy.decide(ctx)
    finally:
        if disable_info_gain:
            policy.INFO_GAIN_VAR_MIN = orig


# ------------------------------------------------------------- simulation ----

def run_policy(policy_fn, profile: str, n_students: int, n_steps: int,
               seed: int) -> dict:
    rng = np.random.default_rng(seed)
    gp = GlobalParams()
    bank = item_bank()
    agg = dict(success=[], mastered=[], uncertainty=[], coverage=[],
               difficulty_first=[], difficulty_second=[], reviews_delivered=[],
               prereq_recovery=[])
    interventions: dict[str, int] = {}
    for _ in range(n_students):
        student = SimStudent.create(profile, rng)
        state = StudentState.from_priors(
            user_id=0, graph_version=1,
            concept_defs=[(CID[c[0]], c[0], 0.0, 1.0) for c in CONCEPTS])
        common = dict(items_by_concept=bank, attempted=set(), recent={},
                      content_shown={}, remediated={}, now=0.0, session_attempts=0)
        correct = attempts = reviews_delivered = 0
        difficulties = []
        theta0 = student.theta.copy()
        weak_idx = np.argsort(theta0)[:3]
        for step in range(n_steps):
            now = float(step) * 3600.0
            common["now"] = now
            out = policy_fn(state, common, rng)
            if out is None:
                continue

            # normalize the decision into an executable step
            if hasattr(out, "intervention"):
                action = out.intervention
                interventions[action] = interventions.get(action, 0) + 1
                if action in ("EXPLAIN", "EXAMPLE", "SIMPLIFY", "BREAK_TASK"):
                    n_prev = common["content_shown"].get(out.concept_id, 0)
                    common["content_shown"][out.concept_id] = n_prev + 1
                    if action in REMEDIATION_ACTIONS:
                        common["remediated"][out.concept_id] = \
                            common["remediated"].get(out.concept_id, 0) + 1
                    student.teach(out.concept_id, n_prev)   # planted instruction effect
                    continue
                if action == "SESSION_WRAP":
                    common["session_attempts"] = 0
                    continue
                if out.concept_id is None or out.item_id is None:
                    continue
                cid = out.concept_id
                item = next(it for it in bank[cid] if it.item_id == out.item_id)
            else:
                action, cid, item = out
                interventions[action] = interventions.get(action, 0) + 1

            if action == "SPACED_REVIEW":
                reviews_delivered += 1

            y = student.respond(cid, item.b, step, rng)
            attempts += 1
            correct += int(y)
            common["attempted"].add(item.item_id)
            common["recent"].setdefault(cid, []).append(
                AttemptRecord(correct=bool(y), latency_ms=6000.0, ts=now))
            common["recent"][cid] = common["recent"][cid][-6:]
            common["remediated"][cid] = 0        # fresh evidence resets remediation budget
            common["session_attempts"] += 1
            difficulties.append(item.b)
            state.apply(Observation(concept_id=cid, correct=bool(y), ts=now,
                                    item_a=item.a, item_b=item.b), gp)

        agg["success"].append(correct / max(attempts, 1))
        agg["mastered"].append(sum(1 for s in state.concepts.values() if s.mastered))
        agg["uncertainty"].append(float(np.sqrt(sum(
            model.eff_var(s, common["now"], gp) for s in state.concepts.values()))))
        agg["coverage"].append(sum(1 for s in state.concepts.values() if s.exposure > 0))
        agg["reviews_delivered"].append(reviews_delivered)
        half = len(difficulties) // 2
        if half > 1:
            agg["difficulty_first"].append(float(np.mean(difficulties[:half])))
            agg["difficulty_second"].append(float(np.mean(difficulties[half:])))
        agg["prereq_recovery"].append(float(np.mean(
            student.theta[weak_idx] - theta0[weak_idx]))
            if profile == "weak_concepts" else 0.0)
    out = {k: float(np.mean(v)) for k, v in agg.items() if v}
    out["interventions"] = interventions
    return out


def run_suite(n_students: int = 200, n_steps: int = 60, seed: int = 11) -> dict:
    import zlib
    results = {}
    policies = {
        "random": lambda s, c, r: policy_random(s, c, r),
        "fixed": lambda s, c, r: policy_fixed(s, c, r),
        "rules_no_gain": lambda s, c, r: policy_rules(s, c, r, disable_info_gain=True),
        "rules_full": lambda s, c, r: policy_rules(s, c, r),
    }
    for pname, fn in policies.items():
        results[pname] = {}
        for profile in PROFILES:
            pseed = seed + zlib.crc32(f"{pname}:{profile}".encode()) % 10000
            results[pname][profile] = run_policy(fn, profile, n_students, n_steps, pseed)
            r = results[pname][profile]
            print(f"{pname:14s} {profile:14s} success={r['success']:.3f} "
                  f"mastered={r['mastered']:.2f} unc={r['uncertainty']:.2f} "
                  f"coverage={r['coverage']:.1f}")
    return results
