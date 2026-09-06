"""End-to-end loop: seed -> diagnostic -> adaptive teaching -> mastery -> traceability.

The client rotates sessions when the policy wraps for fatigue — that is intended
product behavior, not a test workaround.
"""
import os

from fastapi.testclient import TestClient

from alpa.seed.mechanics import ITEMS

CORRECT_BY_STEM = {stem: correct for (_c, _k, stem, _ch, correct, _b, _a) in ITEMS}
QUESTION_WITH_ITEM = {"DIAGNOSTIC", "Q_EASY", "Q_MED", "Q_HARD", "RECALL", "SPACED_REVIEW", "HINT"}


def make_client(tmp_path):
    os.environ["ALPA_DB"] = f"sqlite:///{tmp_path}/test.db"
    from alpa.api.app import create_app
    return TestClient(create_app(os.environ["ALPA_DB"]))


class Loop:
    """Drives the API: serves the next decision, answers item decisions correctly,
    rotates sessions on SESSION_WRAP."""

    def __init__(self, client, user_id):
        self.client, self.user_id = client, user_id
        self.session = client.post(f"/api/users/{user_id}/sessions").json()
        self.teach_seen = []
        self.last_question_trace = None
        self.decision_kinds = set()

    def step(self):
        dec = self.client.get(f"/api/sessions/{self.session['id']}/next").json()
        self.decision_kinds.add(dec["intervention"])
        if dec["intervention"] == "SESSION_WRAP":
            self.session = self.client.post(f"/api/users/{self.user_id}/sessions").json()
            return None
        if dec["intervention"] in ("EXPLAIN", "EXAMPLE"):
            self.teach_seen.append(dec)
            return None  # teaching actions have no attempt (LLM hook, Phase 5)
        item = dec["payload"].get("item")
        if item is None:  # SIMPLIFY / BREAK_TASK / ...: content-side actions
            return None
        res = self.client.post(f"/api/sessions/{self.session['id']}/attempts", json={
            "intervention_id": dec["intervention_id"], "item_id": item["item_id"],
            "choice_index": CORRECT_BY_STEM[item["stem"]], "latency_ms": 6000,
        }).json()
        assert res["correct"] is True
        if dec["intervention"] in QUESTION_WITH_ITEM:
            self.last_question_trace = dec["intervention_id"]
        return res


def test_full_adaptive_loop(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/api/health").json()["llm_coupled"] is False
        user = client.post("/api/users", json={"name": "Ada"}).json()
        loop = Loop(client, user["id"])

        mastery_reached = False
        for _ in range(200):
            res = loop.step()
            if res is not None and res["mastery_after"]["mastered"]:
                mastery_reached = True
                break
        assert "DIAGNOSTIC" in loop.decision_kinds
        assert mastery_reached, "correct answering must eventually produce mastery"

        prog = client.get(f"/api/users/{user['id']}/progress").json()
        assert prog["mastered"] >= 1
        assert prog["total_attempts"] > 8
        vec = next(c for c in prog["concepts"] if c["code"] == "vec_units")
        assert vec["mastered"] is True

        # Traceability: decision joins to model version, inputs, and outcome
        tr = client.get(f"/api/interventions/{loop.last_question_trace}/trace").json()
        assert tr["prediction"]["model_version_id"] >= 1
        assert tr["prediction"]["p_correct"] is not None
        assert tr["prediction"]["features"]["theta_eff"] is not None
        assert tr["outcome"]["correct"] is True
        assert tr["intervention"]["rationale"]
        assert tr["outcome"]["item_id"] == tr["intervention"]["item_id"]


def test_run_to_completion_covers_frontier_and_teach_contract(tmp_path):
    """Answering everything correctly walks the prerequisite frontier across
    concepts; untouched concepts must receive teach actions carrying the
    Phase 5 LLM content contract."""
    with make_client(tmp_path) as client:
        user = client.post("/api/users", json={"name": "Cleo"}).json()
        loop = Loop(client, user["id"])
        for _ in range(400):
            loop.step()
            prog = client.get(f"/api/users/{user['id']}/progress").json()
            if prog["mastered"] >= 4:
                break
        assert prog["mastered"] >= 4
        assert loop.teach_seen, "frontier traversal must teach untouched concepts"
        for dec in loop.teach_seen:
            spec = dec["payload"].get("content_spec")
            assert spec and set(spec) == {"concept", "intervention"}, \
                "teach decisions must expose the LLM generation contract"
            assert dec["payload"].get("generated_content"), \
                "teach decisions must carry provider-generated content"
            gs = dec["payload"].get("generation_spec")
            assert gs and {"concept", "mastery", "uncertainty", "intervention"} <= set(gs), \
                "generation spec must carry the state the ML engine measured"
        assert loop.decision_kinds - QUESTION_WITH_ITEM - {"SESSION_WRAP"}


def test_item_mismatch_is_rejected(tmp_path):
    with make_client(tmp_path) as client:
        user = client.post("/api/users", json={"name": "Bob"}).json()
        session = client.post(f"/api/users/{user['id']}/sessions").json()
        dec = client.get(f"/api/sessions/{session['id']}/next").json()
        item = dec["payload"]["item"]
        r = client.post(f"/api/sessions/{session['id']}/attempts", json={
            "intervention_id": dec["intervention_id"], "item_id": item["item_id"] + 999,
            "choice_index": 0})
        assert r.status_code == 400
