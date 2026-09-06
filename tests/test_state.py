"""Student state: serialization, determinism, mastery/uncertainty separation."""
import copy

from alpa.core.state import Observation, StudentState, states_equal
from alpa.core.types import ConceptSnapshot, GlobalParams

GP = GlobalParams()
NOW = 5_000_000.0


def make_state() -> StudentState:
    st = StudentState.from_priors(user_id=7, graph_version=1, concept_defs=[
        (1, "vec", -0.5, 1.0), (2, "n2", -0.5, 1.0)])
    return st


def test_serialization_roundtrip_exact():
    st = make_state()
    st.apply(Observation(1, True, NOW), GP)
    st.apply(Observation(1, True, NOW + 60), GP)
    st2 = StudentState.from_json(st.to_json())
    assert states_equal(st, st2, tol=0.0)
    assert st.fingerprint() == st2.fingerprint()


def test_transition_is_deterministic():
    obs = [Observation(1, True, NOW + i * 60) for i in range(4)] + \
          [Observation(2, False, NOW + 300)]
    a, b = make_state(), make_state()
    for o in obs:
        a.apply(o, GP)
    for o in obs:
        b.apply(o, GP)
    assert states_equal(a, b, tol=0.0)
    assert a.fingerprint() == b.fingerprint()


def test_order_and_content_sensitivity():
    a = make_state()
    a.apply(Observation(1, True, NOW), GP)
    a.apply(Observation(1, False, NOW + 60), GP)
    b = make_state()
    b.apply(Observation(1, False, NOW), GP)
    b.apply(Observation(1, True, NOW + 60), GP)
    assert not states_equal(a, b)


def test_mastery_and_uncertainty_are_separate():
    st = make_state()
    # drive concept 1 to a strong-but-uncertain estimate
    for i in range(5):
        st.apply(Observation(1, True, NOW + i), GP)
    st.concepts[1].var = 0.8  # uncertainty injected (e.g., long gap, process noise)
    p_m = st.mastery(1, NOW + 5, GP)
    u = st.uncertainty(1, NOW + 5, GP)
    assert p_m > 0.5 and u > 0.8, f"mastery {p_m} / uncertainty {u} must vary independently"
    # mastery high with high uncertainty is a legal, representable state
    st.concepts[1].mastered = True
    assert st.concepts[1].mastered and st.uncertainty(1, NOW + 5, GP) > 0.8


def test_schema_version_rejected():
    st = make_state()
    raw = st.to_json().replace('"schema_version":1', '"schema_version":99')
    try:
        StudentState.from_json(raw)
        raise AssertionError("expected schema rejection")
    except ValueError:
        pass


def test_empty_state_handles_new_concept_queries():
    st = StudentState(user_id=1, graph_version=1)
    assert st.concepts == {}
    assert st.to_json() and StudentState.from_json(st.to_json()).concepts == {}
