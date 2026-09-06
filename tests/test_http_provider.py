"""HTTP LLM provider: real HTTP roundtrips against a mock OpenAI-compatible
server; content generation, defensive parsing, validation gating, and the full
ingest -> approve -> adaptive-session path.

Validation status: VALIDATED against the wire protocol (mock server).
UNVALIDATED against paid external endpoints (no API keys in this environment).
"""
import json
import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from alpa.content.provider import (HTTPChatProvider, GenerationSpec,
                                   ProviderOutputError, parse_json_loose,
                                   register_from_env, REGISTRY)
from tests.mock_llm import EXPECTED_KEY, create_mock_llm

SPEC = GenerationSpec(concept="Torque", mastery=0.42, uncertainty=0.21,
                      difficulty=0.6, intervention="EXAMPLE",
                      prerequisites=("Force",))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def mock_server():
    """Real TCP server speaking the OpenAI-compatible protocol."""
    app = create_mock_llm()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "mock LLM server failed to start"
    yield f"http://127.0.0.1:{port}/v1", app
    server.should_exit = True
    th.join(timeout=5)


@pytest.fixture()
def provider(mock_server):
    base_url, app = mock_server
    p = HTTPChatProvider(base_url=base_url, api_key=EXPECTED_KEY,
                         model="mock-model", name="http:mock-test")
    p._mock = app
    return p


def test_text_generation_roundtrip(provider):
    text = provider.generate_explanation(SPEC)
    assert text.startswith("MOCK EXPLANATION about Torque")
    entry = provider._mock.state.log[-1]
    assert entry["auth"] == f"Bearer {EXPECTED_KEY}"
    assert '"intervention": "EXAMPLE"' in entry["user"], \
        "GenerationSpec must reach the provider verbatim"


def test_question_generation_valid_mcq(provider):
    q = provider.generate_question(SPEC)
    assert q["item_type"] == "multiple_choice" and q["correct_index"] == 0
    assert len(q["choices"]) == 4


def test_question_generation_numerical(provider):
    spec = GenerationSpec(concept="Power", mastery=0.3, uncertainty=0.4,
                          difficulty=0.5, intervention="EXAMPLE")
    q = provider.generate_question(spec)
    assert q["item_type"] == "numerical" and float(q["answer"]) == 250.0


def test_garbled_output_raises(provider):
    spec = GenerationSpec(concept="Garbled", mastery=0.1, uncertainty=0.9,
                          difficulty=0.5, intervention="EXAMPLE")
    with pytest.raises(ProviderOutputError):
        provider.generate_question(spec)


def test_parse_json_loose_variants():
    assert parse_json_loose('{"a": 1}')["a"] == 1
    assert parse_json_loose('noise ```json\n{"a": 2}\n``` tail')["a"] == 2
    assert parse_json_loose('prefix {"a": 3} suffix')["a"] == 3
    with pytest.raises(ProviderOutputError):
        parse_json_loose("no json here")


def test_concept_and_relationship_extraction(provider):
    concepts = provider.extract_concepts("Work and Energy\n\nPower")
    assert [c["name"] for c in concepts] == ["Work and Energy", "Power"]
    rels = provider.extract_relationships([c["name"] for c in concepts], "material")
    assert rels[0]["kind"] == "PREREQUISITE" and rels[0]["src"] == "Power"


# ------------------------------------------------------- full API pipeline --

def make_client(tmp_path, provider_name):
    os.environ["ALPA_DB"] = f"sqlite:///{tmp_path}/httpprov.db"
    from alpa.api.app import create_app
    from alpa import engine
    engine.CONTENT_PROVIDER = provider_name
    return TestClientFactory(create_app(os.environ["ALPA_DB"]))


class TestClientFactory:
    def __init__(self, app):
        from fastapi.testclient import TestClient
        self._cm = TestClient(app)

    def __enter__(self):
        return self._cm.__enter__()

    def __exit__(self, *a):
        return self._cm.__exit__(*a)


def test_full_pipeline_with_http_provider(tmp_path, provider):
    from alpa.content.provider import REGISTRY
    REGISTRY.register(provider)
    with make_client(tmp_path, "http:mock-test") as client:
        # 1) ingest text through the REAL HTTP provider path
        r = client.post("/api/ingest", json={
            "text": "Torque\nPower\nBadConcept\nGarbled",
            "provider": "http:mock-test"}).json()
        assert r["provider"] == "http:mock-test"
        codes = {c["code"] for c in r["concepts"]}
        assert {"torque", "power", "badconcept", "garbled"} <= codes
        types = {i["item_type"] for i in r["items_added"]}
        assert {"multiple_choice", "numerical"} <= types, \
            "valid MCQ and numerical items must be banked"
        assert any("duplicate" in str(rej["errors"]).lower()
                   for rej in r["questions_rejected"]), \
            "ambiguous BadConcept item must be rejected by validation"
        assert any("provider output error" in str(rej["errors"])
                   for rej in r["questions_rejected"]), \
            "garbled provider output must be rejected, not crash the pipeline"
        assert r["edges_added"] == 1

        # 2) LLM-proposed edge is in the graph but NOT traversable until approved
        g = client.get("/api/graph").json()
        llm_edges = [e for e in g["edges"] if e["provenance"].startswith("llm:")]
        assert llm_edges and all(not e["traversable"] for e in llm_edges)
        rid = llm_edges[0]["id"]
        client.post(f"/api/edges/{rid}/approve")
        g = client.get("/api/graph").json()
        assert next(e["traversable"] for e in g["edges"] if e["id"] == rid)

        # 3) generated teaching content in an adaptive session comes from the
        #    provider, with the ML engine's GenerationSpec attached
        user = client.post("/api/users", json={"name": "ada"}).json()
        client.post(f"/api/concepts/{next(c['id'] for c in r['concepts'] if c['code']=='torque')}/approve")
        client.post(f"/api/concepts/{next(c['id'] for c in r['concepts'] if c['code']=='power')}/approve")
        sid = client.post(f"/api/users/{user['id']}/sessions").json()["id"]
        seen_generated = False
        for _ in range(30):
            d = client.get(f"/api/sessions/{sid}/next").json()
            if d["intervention"] in ("EXPLAIN", "EXAMPLE"):
                gc = d["payload"].get("generated_content", "")
                assert gc.startswith("MOCK EXPLANATION"), \
                    "teach content must come from the configured provider"
                assert d["payload"]["content_provider"] == "http:mock-test"
                seen_generated = True
                continue
            item = d["payload"].get("item")
            if item:
                client.post(f"/api/sessions/{sid}/attempts", json={
                    "intervention_id": d["intervention_id"],
                    "item_id": item["item_id"], "choice_index": 0,
                    "latency_ms": 5000})
        assert seen_generated, "session must reach provider-generated teaching"


# ---------------------------------------------------------- env registration

def test_register_from_env_noop_when_unconfigured():
    assert register_from_env(env={}) is None


def test_register_from_env_registers_http_provider():
    name = register_from_env(env={
        "ALPA_LLM_BASE_URL": "http://localhost:11434/v1",
        "ALPA_LLM_API_KEY": "ollama",
        "ALPA_LLM_MODEL": "llama3.1",
        "ALPA_LLM_PROVIDER": "test-env-provider",
    })
    assert name == "test-env-provider"
    assert "test-env-provider" in REGISTRY.names()
    provider = REGISTRY.get("test-env-provider")
    assert isinstance(provider, HTTPChatProvider)
    assert provider.model == "llama3.1"
