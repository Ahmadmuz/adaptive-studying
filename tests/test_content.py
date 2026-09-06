"""Content pipeline: provider abstraction, question validation/grading,
ingestion, and the human-approval gate for LLM-proposed concepts."""
import os

from fastapi.testclient import TestClient

from alpa.content.ingest import clean, extract_text, ingest_text, segment, UnsupportedFormat
from alpa.content.provider import REGISTRY, GenerationSpec, TemplateProvider
from alpa.content.questions import QuestionSpec


def make_client(tmp_path):
    os.environ["ALPA_DB"] = f"sqlite:///{tmp_path}/content.db"
    from alpa.api.app import create_app
    return TestClient(create_app(os.environ["ALPA_DB"]))


SPEC = GenerationSpec(concept="Torque", mastery=0.42, uncertainty=0.21,
                      difficulty=0.6, intervention="EXAMPLE",
                      prerequisites=("Force",))


def test_provider_registry_and_determinism():
    p1, p2 = TemplateProvider(), TemplateProvider()
    assert p1.generate_explanation(SPEC) == p2.generate_explanation(SPEC)
    assert REGISTRY.get("template:v1").name == "template:v1"
    try:
        REGISTRY.get("nonexistent")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_mc_question_validation_and_grading():
    q = QuestionSpec(question="What is torque?", answer="B", explanation="e",
                     concept="Torque", difficulty=0.6, item_type="multiple_choice",
                     grading_method="exact_choice", choices=["A x", "B", "C"],
                     correct_index=1)
    assert q.validate() == []
    assert q.grade("1") == (True, "exact_choice")
    assert q.grade("0") == (False, "exact_choice")


def test_ambiguous_duplicate_choices_rejected():
    q = QuestionSpec(question="Pick one", answer="Same", explanation="",
                     concept="C", difficulty=0.5, item_type="multiple_choice",
                     grading_method="exact_choice",
                     choices=["Same", "same ", "Other"], correct_index=0)
    assert "duplicate/ambiguous choices" in q.validate()


def test_malformed_items_rejected():
    cases = [
        QuestionSpec(question="short", answer="", explanation="", concept="C",
                     difficulty=0.5, item_type="numerical",
                     grading_method="numeric_tolerance", tolerance=0.1),
        QuestionSpec(question="valid question text", answer="3.5", explanation="",
                     concept="C", difficulty=0.5, item_type="numerical",
                     grading_method="exact_choice", tolerance=0.1),
        QuestionSpec(question="valid question text", answer="", explanation="",
                     concept="C", difficulty=1.5, item_type="short_answer",
                     grading_method="normalized_match"),
        QuestionSpec(question="valid question text", answer="", explanation="",
                     concept="C", difficulty=0.5, item_type="structured",
                     grading_method="rubric_external"),
    ]
    for q in cases:
        assert q.validate(), f"expected rejection: {q}"


def test_numerical_and_short_answer_grading():
    qn = QuestionSpec(question="Compute the value", answer="3.5", explanation="",
                      concept="C", difficulty=0.5, item_type="numerical",
                      grading_method="numeric_tolerance", tolerance=0.2)
    assert qn.validate() == []
    assert qn.grade("3.6") == (True, "numeric_tolerance")
    assert qn.grade("3.8") == (False, "numeric_tolerance")
    qs = QuestionSpec(question="Name the law", answer="Newton's Second Law",
                      explanation="", concept="C", difficulty=0.5,
                      item_type="short_answer", grading_method="normalized_match")
    assert qs.grade("newton's   second law") == (True, "normalized_match")


def test_ingest_text_pipeline():
    text = "Work and Energy\n\nWork is force times displacement.\n\nPower\n\n" \
           "Power is the rate of doing work."
    rep = ingest_text(text, TemplateProvider(), source="unit-test")
    assert rep.n_segments >= 1
    names = [c["name"] for c in rep.concept_proposals]
    assert "Work and Energy" in names and "Power" in names
    assert all(c["provenance"].startswith("llm:") for c in rep.concept_proposals)
    assert all(c["human_reviewed"] is False for c in rep.concept_proposals)
    assert rep.questions_accepted, "template provider must yield valid MCQ"


def test_extract_text_formats_and_errors():
    assert "hello" in extract_text(b"hello world", "notes.txt")
    try:
        extract_text(b"x", "file.xyz")
        raise AssertionError("expected UnsupportedFormat")
    except UnsupportedFormat:
        pass
    assert clean("a   b\n\n\n\nc") == "a b\n\nc"
    assert len(segment("para\n\n" + "x" * 3000)) >= 2


def test_approval_gate_end_to_end(tmp_path):
    with make_client(tmp_path) as client:
        r = client.post("/api/ingest", json={
            "topics": ["Work and Energy", "Power"], "provider": "template:v1"}).json()
        assert len(r["concepts"]) == 2
        assert r["items_added"], "validated MCQ items must reach the bank"
        assert all(c["human_reviewed"] is False for c in r["concepts"])

        user = client.post("/api/users", json={"name": "rev"}).json()
        prog = client.get(f"/api/users/{user['id']}/progress").json()
        codes = [c["code"] for c in prog["concepts"]]
        assert "work_and_energy" not in codes, \
            "unreviewed LLM concepts must not enter the adaptive loop"

        cid = r["concepts"][0]["id"]
        assert client.post(f"/api/concepts/{cid}/approve").json()["human_reviewed"] is True
        prog = client.get(f"/api/users/{user['id']}/progress").json()
        codes = [c["code"] for c in prog["concepts"]]
        assert "work_and_energy" in codes, "approved concept must become teachable"

        g = client.get("/api/graph").json()
        node = next(n for n in g["nodes"] if n["code"] == "work_and_energy")
        assert node["id"] == cid

        # regression: items attached to UNAPPROVED concepts must never be
        # served (defect found in live smoke: diagnostic picked a Power item)
        unapproved = r["concepts"][1]["id"]
        sid = client.post(f"/api/users/{user['id']}/sessions").json()
        for _ in range(12):
            d = client.get(f"/api/sessions/{sid['id']}/next").json()
            assert d["concept_id"] != unapproved, \
                "unapproved concept leaked into served decisions"
            item = d["payload"].get("item")
            if item:
                client.post(f"/api/sessions/{sid['id']}/attempts", json={
                    "intervention_id": d["intervention_id"],
                    "item_id": item["item_id"], "choice_index": 0,
                    "latency_ms": 4000})
