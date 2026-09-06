"""Unit tests for the hybrid student model (Phase 1 §3.5 math)."""
from alpa.core import model
from alpa.core.types import ConceptSnapshot, GlobalParams, ItemSpec

GP = GlobalParams()
NOW = 1_000_000.0
DAY = 86400.0


def snap(**kw) -> ConceptSnapshot:
    base = dict(concept_id=1, code="c1", theta=0.0, var=1.0)
    base.update(kw)
    return ConceptSnapshot(**base)


ITEM = ItemSpec(item_id=1, stem="s", choices=("a", "b"), correct_index=0,
                concept_ids=(1,), b=0.0, a=1.0)


def test_sigmoid_and_cdf():
    assert abs(model.sigmoid(0.0) - 0.5) < 1e-12
    assert model.sigmoid(10.0) > 0.999
    assert abs(model.normal_cdf(0.0) - 0.5) < 1e-12


def test_prediction_monotone_in_ability():
    p_low = model.p_correct(snap(theta=-1.0), ITEM, NOW, GP)
    p_high = model.p_correct(snap(theta=1.0), ITEM, NOW, GP)
    assert p_low < 0.5 < p_high


def test_successes_raise_and_failures_lower_prediction():
    s = snap(succ=3)
    f = snap(fail=3)
    assert model.p_correct(s, ITEM, NOW, GP) > 0.5
    assert model.p_correct(f, ITEM, NOW, GP) < 0.5


def test_retention_decays_and_stability_buffers():
    fresh = snap(last_success_ts=NOW - DAY, stability=1.0)
    assert model.retention(fresh, NOW, GP) < 1.0
    stable = snap(last_success_ts=NOW - DAY, stability=20.0)
    assert model.retention(stable, NOW, GP) > model.retention(fresh, NOW, GP)
    assert model.retention(snap(), NOW, GP) == 1.0  # never learned -> nothing to decay


def test_forgetting_lowers_effective_ability():
    learned = snap(theta=1.0, last_success_ts=NOW - 30 * DAY, stability=1.0)
    assert model.eff_theta(learned, NOW, GP) < 1.0
    assert model.p_correct(learned, ITEM, NOW, GP) < model.p_correct(snap(theta=1.0), ITEM, NOW, GP)


def test_update_direction_and_variance_shrinkage():
    s0 = snap()
    s_win = model.update(s0, ITEM, 1, NOW, GP)
    s_lose = model.update(s0, ITEM, 0, NOW, GP)
    assert s_win.theta > s0.theta > s_lose.theta
    assert s_win.var < s0.var and s_lose.var < s0.var
    assert s_win.succ == 1 and s_lose.fail == 1
    assert s_win.last_success_ts == NOW
    assert s_win.stability > s0.stability > s_lose.stability
    assert s_win.exposure == 1


def test_repeated_success_approaches_mastery():
    """Under difficulty-adaptive serving (the policy keeps p ~ 0.55), mastery is
    reachable in ~8-10 correct attempts — the system's core convergence promise."""
    import math
    s = snap()
    t = NOW
    for _ in range(8):
        feats = model.predict(s, ITEM, t, GP)
        z = math.log(0.55 / 0.45)
        it = ItemSpec(item_id=9, stem="s", choices=("a", "b"), correct_index=0,
                      concept_ids=(1,),
                      b=feats["theta_eff"] + feats["succ_term"] + feats["fail_term"] - z, a=1.0)
        s = model.update(s, it, 1, t, GP)
        t += 3600
    assert model.mastery_prob(s, t, GP) > 0.9


def test_mastery_hysteresis():
    s = snap(theta=1.2, var=0.05, mastered=False)
    assert model.apply_mastery_hysteresis(s, NOW, GP) is True
    # demotion requires crossing the lower threshold — borderline states stick
    s2 = snap(theta=1.2, var=0.05, mastered=True)
    assert model.apply_mastery_hysteresis(s2, NOW, GP) is True
    weak = snap(theta=0.0, var=0.05, mastered=True)
    assert model.apply_mastery_hysteresis(weak, NOW, GP) is False


def test_uncertainty_grows_while_unseen():
    s = snap(var=0.1, last_update_ts=NOW - 30 * DAY)
    assert model.eff_var(s, NOW, GP) > 0.1


def test_review_due_days():
    s = snap(last_success_ts=NOW, stability=5.0)
    due = model.review_due_days(s, NOW, GP)
    assert due is not None and due > 0
    assert model.review_due_days(snap(), NOW, GP) is None
    # decayed retention -> due now
    old = snap(last_success_ts=NOW - 200 * DAY, stability=1.0)
    assert model.review_due_days(old, NOW, GP) == 0.0
