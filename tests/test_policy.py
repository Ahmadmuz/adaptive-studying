"""Policy rule-table tests: each rule cell fires under its specified conditions."""
import random

from alpa.core.policy import (
    PolicyContext, decide, EXPLAIN, EXAMPLE, Q_EASY, Q_MED, HINT,
    SIMPLIFY, BREAK_TASK, SPACED_REVIEW, SESSION_WRAP, FATIGUE_MAX_ATTEMPTS,
)
from alpa.core.types import AttemptRecord, ConceptSnapshot, GlobalParams, ItemSpec

GP = GlobalParams()
NOW = 2_000_000.0
DAY = 86400.0


def item(iid: int, cid: int, b: float, a: float = 1.0, kind="practice") -> ItemSpec:
    return ItemSpec(item_id=iid, stem=f"q{iid}", choices=("a", "b"), correct_index=0,
                    concept_ids=(cid,), b=b, a=a, kind=kind)


def snap(cid: int, **kw) -> ConceptSnapshot:
    base = dict(concept_id=cid, code=f"c{cid}", theta=-0.5, var=1.0)
    base.update(kw)
    return ConceptSnapshot(**base)


def ctx(snapshots, items, epsilon=0.0, **kw) -> PolicyContext:
    ids = sorted(snapshots)
    defaults = dict(
        snapshots=snapshots, topo_order=ids, prereqs={i: [] for i in ids},
        items_by_concept={}, attempted_item_ids=set(), recent_attempts={},
        content_shown={}, now=NOW, session_attempts=0, gp=GP,
        rng=random.Random(7), epsilon=epsilon,
    )
    defaults.update(kw)
    ibc = defaults["items_by_concept"]
    for it in items:
        ibc.setdefault(it.concept_ids[0], []).append(it)
    return PolicyContext(**defaults)


def test_fresh_student_gets_explanation():
    c = ctx({1: snap(1)}, [item(10, 1, b=0.0)])
    d = decide(c)
    assert d.intervention == EXPLAIN and d.concept_id == 1 and d.item_id is None


def test_second_teaching_action_is_example():
    c = ctx({1: snap(1)}, [item(10, 1, b=0.0)], content_shown={1: 1})
    assert decide(c).intervention == EXAMPLE


def test_medium_band_gets_medium_question():
    c = ctx({1: snap(1, theta=0.0, exposure=2)}, [item(10, 1, b=0.0)], content_shown={1: 2})
    d = decide(c)
    assert d.intervention == Q_MED and d.item_id == 10 and d.target_p is not None


def test_frontier_respects_topological_order():
    c = ctx({1: snap(1, mastered=True), 2: snap(2)},
            [item(10, 1, b=0.0), item(11, 2, b=0.0)],
            prereqs={1: [], 2: [1]})
    d = decide(c)
    assert d.concept_id == 2 and d.intervention == EXPLAIN


def test_low_retention_triggers_spaced_review():
    old = NOW - 60 * DAY
    c = ctx({1: snap(1, mastered=True, last_success_ts=old, stability=1.0),
             2: snap(2)},
            [item(10, 1, b=0.0), item(11, 2, b=0.0)],
            prereqs={1: [], 2: [1]}, attempted_item_ids={10})
    d = decide(c)
    assert d.intervention == SPACED_REVIEW and d.concept_id == 1 and d.item_id == 10


def test_misconception_streak_reduces_load():
    recents = [AttemptRecord(correct=False, latency_ms=9000, ts=NOW - i) for i in (3, 2, 1)]
    c = ctx({1: snap(1, exposure=5, fail=5)}, [item(10, 1, b=0.0)],
            recent_attempts={1: recents})
    assert decide(c).intervention in (SIMPLIFY, BREAK_TASK)


def test_short_history_does_not_trigger_remediation():
    """Misconception must not fire on bare streaks without sustained evidence —
    the over-firing defect measured in synthetic policy evaluation."""
    recents = [AttemptRecord(correct=False, latency_ms=9000, ts=NOW - i) for i in (3, 2, 1)]
    c = ctx({1: snap(1, exposure=2, fail=2)}, [item(10, 1, b=0.0)],
            recent_attempts={1: recents})
    assert decide(c).intervention not in (SIMPLIFY, BREAK_TASK)


def test_remediation_does_not_loop_without_new_evidence():
    """After one remediation with no fresh attempt, the policy must serve an
    item (easier band), not repeat remediation — the identical-evidence loop
    defect found in policy_eval."""
    recents = [AttemptRecord(correct=False, latency_ms=9000, ts=NOW - i) for i in (3, 2, 1)]
    s = snap(1, exposure=5, fail=5)
    c = ctx({1: s}, [item(10, 1, b=-0.8), item(11, 1, b=-0.2)],
            recent_attempts={1: recents})
    d1 = decide(c)
    assert d1.intervention in (SIMPLIFY, BREAK_TASK)
    c2 = ctx({1: s}, [item(10, 1, b=-0.8), item(11, 1, b=-0.2)],
             recent_attempts={1: recents}, remediated={1: 1})
    d2 = decide(c2)
    assert d2.intervention not in (SIMPLIFY, BREAK_TASK)
    assert d2.item_id is not None, "must fall through to an actionable item"


def test_procedural_slip_gets_hint_not_reteach():
    recents = [AttemptRecord(correct=True, latency_ms=8000, ts=NOW - 2),
               AttemptRecord(correct=False, latency_ms=2000, ts=NOW - 1)]
    c = ctx({1: snap(1, exposure=2, succ=2, fail=1)}, [item(10, 1, b=0.0)],
            recent_attempts={1: recents}, attempted_item_ids={10})
    d = decide(c)
    assert d.intervention == HINT and d.item_id == 10


def test_fatigue_wraps_session():
    c = ctx({1: snap(1)}, [item(10, 1, b=0.0)], session_attempts=FATIGUE_MAX_ATTEMPTS)
    assert decide(c).intervention == SESSION_WRAP


def test_exploration_is_flagged_and_safe():
    class AlwaysZero(random.Random):
        def random(self):
            return 0.0
    c = ctx({1: snap(1, theta=0.0, exposure=2)}, [item(10, 1, b=0.0), item(11, 1, b=-0.3)],
            content_shown={1: 2}, epsilon=1.0, rng=AlwaysZero(1))
    d = decide(c)
    assert d.exploration is True and d.intervention in (Q_EASY, Q_MED)


def test_deterministic_without_exploration():
    mk = lambda: ctx({1: snap(1, theta=0.0, exposure=2)}, [item(10, 1, b=0.0), item(11, 1, b=0.6)])
    assert decide(mk()).intervention == decide(mk()).intervention


def test_explain_payload_carries_llm_contract():
    d = decide(ctx({1: snap(1)}, [item(10, 1, b=0.0)]))
    assert d.payload["content_spec"] == {"concept": "c1", "intervention": EXPLAIN}


def test_information_gain_resolves_borderline_mastery():
    """Borderline mastery probability + high variance => max-information item,
    rationale tagged info_gain; band selection otherwise."""
    from alpa.core.policy import INFO_GAIN_P_LO, INFO_GAIN_P_HI
    from alpa.core import model as m
    # theta/var chosen so p_mastery sits inside the info-gain band
    s = snap(1, theta=0.75, var=0.45, exposure=3, succ=2, fail=1)
    assert INFO_GAIN_P_LO <= m.mastery_prob(s, NOW, GP) < INFO_GAIN_P_HI
    items = [item(10, 1, b=-0.5), item(11, 1, b=0.7), item(12, 1, b=0.75)]
    d = decide(ctx({1: s}, items, content_shown={1: 2}))
    assert d.intervention == Q_MED
    assert d.rationale.startswith("info_gain:resolve_mastery")
    # must be the max-Fisher item: predicted p closest to 0.5 among candidates
    ps = {it.item_id: m.p_correct(s, it, NOW, GP) for it in items}
    assert d.item_id == min(ps, key=lambda k: abs(ps[k] - 0.5))
