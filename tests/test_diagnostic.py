"""Cold-start diagnostic: max-information selection and budget stopping."""
from alpa.core.diagnostic import diagnostic_budget, fisher_information, pick_diagnostic_item
from alpa.core.types import ConceptSnapshot, GlobalParams, ItemSpec

GP = GlobalParams()
NOW = 3_000_000.0


def snap(cid, theta):
    return ConceptSnapshot(concept_id=cid, code=f"c{cid}", theta=theta, var=1.0)


def test_information_peaks_where_difficulty_matches_ability():
    assert fisher_information(1.0, 0.5) > fisher_information(1.0, 0.2)
    assert fisher_information(2.0, 0.5) > fisher_information(1.0, 0.5)


def test_picks_item_matched_to_current_ability():
    snaps = {1: snap(1, theta=-0.5)}
    items = [
        ItemSpec(1, "s", ("a",), 0, (1,), b=-1.6, a=1.0),
        ItemSpec(2, "s", ("a",), 0, (1,), b=-0.5, a=1.0),
        ItemSpec(3, "s", ("a",), 0, (1,), b=1.5, a=1.0),
    ]
    picked, p = pick_diagnostic_item(snaps, items, set(), lambda cid: snaps[cid], NOW, GP)
    assert picked.item_id == 2
    assert 0.35 < p < 0.65  # near maximum-information operating point


def test_excludes_attempted_items():
    snaps = {1: snap(1, theta=-0.5)}
    items = [ItemSpec(2, "s", ("a",), 0, (1,), b=-0.5, a=1.0)]
    assert pick_diagnostic_item(snaps, items, {2}, lambda cid: snaps[cid], NOW, GP) is None


def test_prefers_diagnostic_kind():
    snaps = {1: snap(1, theta=0.0)}
    items = [
        ItemSpec(1, "s", ("a",), 0, (1,), b=0.0, a=1.0, kind="practice"),
        ItemSpec(2, "s", ("a",), 0, (1,), b=0.1, a=1.0, kind="diagnostic"),
    ]
    picked, _ = pick_diagnostic_item(snaps, items, set(), lambda cid: snaps[cid], NOW, GP)
    assert picked.kind == "diagnostic"


def test_budget_stopping_rule():
    assert diagnostic_budget(0) is True
    assert diagnostic_budget(7) is True
    assert diagnostic_budget(8) is False
