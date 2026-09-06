"""Pure data types shared by the ML core. No DB, no IO — keep it that way."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GlobalParams:
    """Population-level parameters of the global model (frozen per model version).

    Every field has a named interpretation (Phase 1, §3.5). The online path never
    writes these; they are refit offline on snapshots and release-gated.
    """

    alpha: float = 0.30          # power-law forgetting exponent, in (0, 1]
    eta: float = 0.60            # log-retention -> effective-ability shift
    beta: float = 0.15           # PFA benefit per prior success (auxiliary: theta
                                 # already absorbs practice via the Bayesian update;
                                 # large beta double-counts evidence and stalls mastery)
    delta: float = 0.25          # PFA penalty per prior failure
    rho: float = 0.01            # process noise: posterior variance added per day unseen
    theta_mastery: float = 0.6   # MVP anchor (~65% on median items, ~83% on easy ones).
                                 # Re-anchor against human benchmark items with real data.
    tau_enter: float = 0.85      # P(mastery) to promote
    tau_exit: float = 0.80       # P(mastery) to demote (hysteresis)
    stability_growth: float = 1.4   # multiplicative stability gain per success
    stability_decay: float = 0.6    # multiplicative stability loss per failure
    stability_min: float = 0.5
    stability_max: float = 120.0    # days
    succ_cap: int = 8            # PFA count caps (calibration-drift guard)
    fail_cap: int = 8


@dataclass(frozen=True)
class ItemSpec:
    """An assessment item. Multi-KC: `concept_ids[0]` is the primary concept."""

    item_id: int
    stem: str
    choices: tuple[str, ...]
    correct_index: int
    concept_ids: tuple[int, ...]
    b: float                     # difficulty prior on logit scale (becomes empirical IRT b)
    a: float = 1.0               # discrimination
    kind: str = "practice"       # diagnostic | practice | retrieval_probe | transfer
    item_type: str = "multiple_choice"   # multiple_choice | numerical | short_answer | structured


@dataclass
class ConceptSnapshot:
    """Individual student state for one concept (the individual half of the
    global/individual separation). Mutable by design: model.update returns a new one."""

    concept_id: int
    code: str
    theta: float                 # point estimate of ability (logit units)
    var: float                   # posterior variance (uncertainty)
    succ: int = 0
    fail: int = 0
    stability: float = 1.0       # memory stability, days (grows with successful retrieval)
    last_success_ts: float | None = None
    last_update_ts: float | None = None
    mastered: bool = False       # mastery is a *decision* (hysteresis), not a score
    exposure: int = 0


@dataclass(frozen=True)
class AttemptRecord:
    """Telemetry for one past attempt, as consumed by the policy."""

    correct: bool
    latency_ms: float | None
    ts: float
    paste_detected: bool = False


@dataclass
class Decision:
    """Output of the intervention policy pi(state) — the exact contract handed to
    content generation (Phase 5) and logged for audit."""

    intervention: str
    concept_id: int | None
    item_id: int | None
    target_p: float | None       # desired P(correct) of the chosen item
    rationale: str                 # rule-cell id / selection criterion
    exploration: bool = False
    payload: dict = field(default_factory=dict)


# Intervention vocabulary (Phase 1, §3.7). Strings are the wire format; do not rename.
EXPLAIN = "EXPLAIN"
EXAMPLE = "EXAMPLE"
Q_EASY = "Q_EASY"
Q_MED = "Q_MED"
Q_HARD = "Q_HARD"
RECALL = "RECALL"
SPACED_REVIEW = "SPACED_REVIEW"
HINT = "HINT"
SIMPLIFY = "SIMPLIFY"
CHANGE_REPRESENTATION = "CHANGE_REPRESENTATION"
BREAK_TASK = "BREAK_TASK"
REVISIT_PREREQUISITE = "REVISIT_PREREQUISITE"
DIAGNOSTIC = "DIAGNOSTIC"
SESSION_WRAP = "SESSION_WRAP"
FLASHCARD = "FLASHCARD"
RETRIEVAL_PRACTICE = "RETRIEVAL_PRACTICE"
SUMMARY = "SUMMARY"
MINI_LESSON = "MINI_LESSON"

QUESTION_ACTIONS = {DIAGNOSTIC, Q_EASY, Q_MED, Q_HARD, RECALL, SPACED_REVIEW, HINT}
TEACH_ACTIONS = (EXPLAIN, EXAMPLE, SIMPLIFY, BREAK_TASK, SUMMARY, MINI_LESSON)
