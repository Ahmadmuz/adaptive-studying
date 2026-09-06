"""Intervention policy pi(student_state, concept_state, context) — Phase 1 §3.7.

MVP form: deterministic rule table over (P(correct), uncertainty/error pattern,
retention, session fatigue), wrapped by epsilon-exploration over a restricted safe
action set. Every decision carries a `rationale` (rule-cell id) so logged data is
usable for off-policy evaluation later.

The policy is a pure function over plain data; the engine assembles the context.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import model
from .types import (
    EXPLAIN, EXAMPLE, Q_EASY, Q_MED, Q_HARD, SPACED_REVIEW, HINT,
    SIMPLIFY, BREAK_TASK, REVISIT_PREREQUISITE, SESSION_WRAP,
    FLASHCARD, RETRIEVAL_PRACTICE, SUMMARY, MINI_LESSON,
    AttemptRecord, ConceptSnapshot, Decision, GlobalParams, ItemSpec,
)

# --- tunables (policy-version parameters; logged with every decision) ---------
REVIEW_THRESHOLD = 0.60        # predicted retention below which review is due
SLIP_LATENCY_MS = 5000.0       # fast-and-wrong window for procedural-slip heuristic
MISCONCEPTION_STREAK = 3       # consecutive failures => misconception branch
PREREQ_WEAK_P = 0.60           # prereq mastery prob below which we divert
MISCONCEPTION_MIN_EXPOSURE = 4 # remediation only after meaningful exposure
TARGET_P_EASY = 0.70           # desired P(correct) per difficulty band
TARGET_P_MED = 0.55
TARGET_P_HARD = 0.30
FATIGUE_MAX_ATTEMPTS = 25
EXPLORATION_EPSILON = 0.05
INFO_GAIN_P_LO = 0.55          # mastery-uncertainty band: not yet promotable but
INFO_GAIN_P_HI = 0.85          # close — spend one max-information item to resolve
INFO_GAIN_VAR_MIN = 0.30       # posterior variance (sd^2) that qualifies


@dataclass
class PolicyContext:
    snapshots: dict[int, ConceptSnapshot]                 # concept_id -> state
    topo_order: list[int]                                 # concept ids in prerequisite order
    prereqs: dict[int, list[int]]
    items_by_concept: dict[int, list[ItemSpec]]
    attempted_item_ids: set[int]
    recent_attempts: dict[int, list[AttemptRecord]]       # last attempts per concept (asc ts)
    content_shown: dict[int, int]                         # EXPLAIN/EXAMPLE deliveries per concept
    now: float
    session_attempts: int
    gp: GlobalParams
    rng: random.Random = field(default_factory=random.Random)
    epsilon: float = EXPLORATION_EPSILON
    remediated: dict[int, int] = field(default_factory=dict)
    # ^ remediation actions delivered since the concept's last attempt.
    # Prevents identical-evidence remediation loops (measured defect in
    # policy_eval); the caller owns incrementing/resetting this counter.


def _pick_item(
    ctx: PolicyContext, concept_id: int, target_p: float, snap: ConceptSnapshot, allow_reuse: bool = False
) -> tuple[ItemSpec | None, float | None]:
    """Item selection: difficulty band is the primary criterion (closest to
    target_p); within near-ties, Fisher information breaks ties — i.e., prefer
    the item that keeps P(correct) near target while reducing uncertainty
    fastest. Information gain never overrides the pedagogical band."""
    cands = ctx.items_by_concept.get(concept_id, [])
    fresh = [it for it in cands if it.item_id not in ctx.attempted_item_ids]
    pool = fresh or (cands if allow_reuse else [])
    if not pool:
        return None, None
    scored = []
    for it in pool:
        p = model.p_correct(snap, it, ctx.now, ctx.gp)
        info = it.a * it.a * p * (1.0 - p)
        scored.append((abs(p - target_p), -info, p, it))
    scored.sort(key=lambda t: (t[0], t[1]))
    _, _, p, item = scored[0]
    return item, p


def _rep_p(ctx: PolicyContext, concept_id: int, snap: ConceptSnapshot) -> tuple[float, list[ItemSpec]]:
    """Representative P(correct): prediction on the item closest to the MED target."""
    cands = [it for it in ctx.items_by_concept.get(concept_id, [])
             if it.item_id not in ctx.attempted_item_ids] or ctx.items_by_concept.get(concept_id, [])
    if not cands:
        return 0.0, []
    scored = sorted(
        ((abs(model.p_correct(snap, it, ctx.now, ctx.gp) - TARGET_P_MED),
          model.p_correct(snap, it, ctx.now, ctx.gp)) for it in cands),
        key=lambda t: t[0],
    )
    return scored[0][1], cands


def _frontier_concept(ctx: PolicyContext) -> int | None:
    """First unmastered concept (topological order) whose prereqs are all mastered."""
    for cid in ctx.topo_order:
        snap = ctx.snapshots.get(cid)
        if snap is not None and snap.mastered:
            continue
        if all(ctx.snapshots[p].mastered for p in ctx.prereqs.get(cid, []) if p in ctx.snapshots):
            return cid
    return None


def _weakest_prereq(ctx: PolicyContext, concept_id: int) -> int | None:
    weak = [(model.mastery_prob(ctx.snapshots[p], ctx.now, ctx.gp), p)
            for p in ctx.prereqs.get(concept_id, []) if p in ctx.snapshots
            and ctx.snapshots[p].exposure > 0]
    if not weak:
        return None
    p_m, pid = min(weak)
    return pid if p_m < PREREQ_WEAK_P else None


def _classify_error_pattern(ctx: PolicyContext, concept_id: int) -> str:
    """Heuristic error taxonomy from telemetry. Returns one of:
    none | procedural_slip | misconception. (Gaming detection: Phase 2.5.)
    Misconception requires SUSTAINED evidence — a bare streak over-fires on
    noisy learners (measured in policy_eval; the gate below is the fix)."""
    recent = ctx.recent_attempts.get(concept_id, [])
    if not recent:
        return "none"
    last = recent[-1]
    snap = ctx.snapshots[concept_id]
    if not last.correct and last.latency_ms is not None and last.latency_ms < SLIP_LATENCY_MS \
            and snap.succ > snap.fail:
        return "procedural_slip"
    window = recent[-5:]
    fail_rate = sum(1 for a in window if not a.correct) / len(window)
    streak_fail = len(recent) >= MISCONCEPTION_STREAK and \
        all(not a.correct for a in recent[-MISCONCEPTION_STREAK:])
    if streak_fail and snap.exposure >= MISCONCEPTION_MIN_EXPOSURE and fail_rate >= 0.5:
        return "misconception"
    return "none"


def _spaced_review(ctx: PolicyContext) -> Decision | None:
    """Mastered concept whose predicted retention dropped below threshold."""
    cands = []
    for cid, snap in ctx.snapshots.items():
        if not snap.mastered or snap.last_success_ts is None:
            continue
        r = model.retention(snap, ctx.now, ctx.gp)
        if r < REVIEW_THRESHOLD:
            cands.append((r, cid))
    if not cands:
        return None
    _, cid = min(cands)
    snap = ctx.snapshots[cid]
    item, p = _pick_item(ctx, cid, target_p=0.5, snap=snap, allow_reuse=True)  # retrieval probe
    if item is None:
        return None
    return Decision(
        intervention=SPACED_REVIEW, concept_id=cid, item_id=item.item_id, target_p=p,
        rationale="spaced_review:retention<%.2f" % REVIEW_THRESHOLD,
        payload={"predicted_retention": model.retention(snap, ctx.now, ctx.gp),
                 "note": "Retrieval practice: answer without hints."},
    )


def decide(ctx: PolicyContext) -> Decision:
    """The rule table. Order = priority; first matching cell wins."""

    # 0. session fatigue — protect return rate, not just correctness
    if ctx.session_attempts >= FATIGUE_MAX_ATTEMPTS:
        return Decision(SESSION_WRAP, None, None, None, "fatigue:max_attempts",
                        payload={"note": "Session wrap: schedule spaced reviews for next session."})

    # 1. retention review outranks new content (testing effect + spacing)
    review = _spaced_review(ctx)
    if review is not None:
        return review

    # 2. frontier concept (first unmastered with all prereqs mastered)
    cid = _frontier_concept(ctx)
    if cid is None:
        return Decision(SESSION_WRAP, None, None, None, "frontier:all_mastered",
                        payload={"note": "All concepts mastered; schedule reviews."})
    snap = ctx.snapshots[cid]
    recent = ctx.recent_attempts.get(cid, [])
    shown = ctx.content_shown.get(cid, 0)
    error = _classify_error_pattern(ctx, cid)

    # 2a. repeated failure on concept with a weak prereq -> go down the graph
    if error == "misconception" and len(recent) >= MISCONCEPTION_STREAK:
        weak_p = _weakest_prereq(ctx, cid)
        if weak_p is not None:
            item, p = _pick_item(ctx, weak_p, TARGET_P_MED, ctx.snapshots[weak_p], allow_reuse=True)
            return Decision(REVISIT_PREREQUISITE, weak_p, item.item_id if item else None, p,
                            "prereq:weak_after_streak",
                            payload={"from_concept": cid})

    # 2b. procedural slip: fast wrong answer after a history of success -> hint, retry
    if error == "procedural_slip":
        item, p = _pick_item(ctx, cid, TARGET_P_MED, snap, allow_reuse=True)
        return Decision(HINT, cid, item.item_id if item else None, p, "error:procedural_slip",
                        payload={"note": "Fast incorrect response with prior success: hint, then retry."})

    # 2c. misconception without weak prereq -> reduce load / decompose
    # (alternating). At most ONE remediation per concept without fresh attempt
    # evidence; after that the band logic serves an easier item instead.
    if error == "misconception" and ctx.remediated.get(cid, 0) < 1:
        action = SIMPLIFY if (shown + len(recent)) % 2 == 0 else BREAK_TASK
        return Decision(action, cid, None, None, "error:misconception",
                        payload={"note": "Consecutive failures: simplify the presentation / break the task."})

    # 2c'. information gain: mastery decision is pending (borderline p_m) and
    # uncertainty is high -> serve the maximum-Fisher-information item. One such
    # item resolves the mastery decision faster than any fixed-band choice.
    if snap.exposure > 0 and snap.var > INFO_GAIN_VAR_MIN:
        p_m = model.mastery_prob(snap, ctx.now, ctx.gp)
        if INFO_GAIN_P_LO <= p_m < INFO_GAIN_P_HI:
            cands = [it for it in ctx.items_by_concept.get(cid, [])
                     if it.item_id not in ctx.attempted_item_ids] \
                or ctx.items_by_concept.get(cid, [])
            if cands:
                scored = sorted(
                    ((-(it.a ** 2 * (p := model.p_correct(snap, it, ctx.now, ctx.gp))
                       * (1.0 - p)), p, it) for it in cands))
                _, p, best = scored[0]
                return Decision(Q_MED, cid, best.item_id, p,
                                f"info_gain:resolve_mastery:p_m={p_m:.2f}")

    # 2d. unseen concept -> teach before testing (worked-example superiority for novices)
    # Rotate through EXPLAIN, EXAMPLE, SUMMARY, MINI_LESSON for variety
    if snap.exposure == 0 and shown < 4:
        teach_sequence = [EXPLAIN, EXAMPLE, SUMMARY, MINI_LESSON]
        action = teach_sequence[shown % len(teach_sequence)]
        return Decision(action, cid, None, None, f"teach:{action.lower()}",
                        payload={"note": "LLM generation hook (Phase 5).",
                                 "content_spec": {"concept": snap.code, "intervention": action}})

    # 2e. difficulty bands on representative P(correct)
    rep_p, _ = _rep_p(ctx, cid, snap)
    if rep_p < 0.25 and shown < 4:
        teach_sequence = [EXPLAIN, EXAMPLE, SUMMARY, MINI_LESSON]
        action = teach_sequence[shown % len(teach_sequence)]
        return Decision(action, cid, None, None, f"teach:low_p:{action.lower()}",
                        payload={"rep_p": rep_p,
                                 "content_spec": {"concept": snap.code, "intervention": action}})

    if rep_p < 0.45:
        band, target, action = "easy", TARGET_P_EASY, Q_EASY
    elif rep_p <= 0.80:
        band, target, action = "medium", TARGET_P_MED, Q_MED
    else:
        band, target, action = "hard", TARGET_P_HARD, Q_HARD

    # allow_reuse: re-serving known items is legitimate retrieval practice and is
    # the only way to keep measuring once the fresh-item pool is exhausted.
    item, p = _pick_item(ctx, cid, target, snap, allow_reuse=True)
    if item is None:
        # concept has no items at all (data bug): never spin on a teach loop
        return Decision(SESSION_WRAP, cid, None, None, "teach:concept_has_no_items",
                        payload={"rep_p": rep_p})
    decision = Decision(action, cid, item.item_id, p, f"band:{band}:p={rep_p:.2f}")

    # 3. exploration wrapper (safe actions only; logged with exploration=True)
    if ctx.rng.random() < ctx.epsilon:
        alt_item, alt_p = _pick_item(ctx, cid, TARGET_P_EASY if action == Q_MED else TARGET_P_MED, snap)
        if alt_item is not None:
            return Decision(
                Q_EASY if action == Q_MED else Q_MED, cid, alt_item.item_id, alt_p,
                f"exploration:from_{action}", exploration=True,
            )
    return decision
