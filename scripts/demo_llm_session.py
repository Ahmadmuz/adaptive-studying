#!/usr/bin/env python3
"""Live demonstration of the NEXT-block pipeline, all over real TCP:

    PDF bytes -> ingestion (LLM provider) -> concept graph with LLM-proposed
    edges -> human approval gates -> adaptive session with provider-generated
    teaching content -> deterministic grading of generated numerical items ->
    decision trace.

The LLM endpoint is the deterministic mock OpenAI-compatible server
(tests/mock_llm.py): this validates the WIRE PROTOCOL and the pipeline; it is
NOT validation of any specific commercial model.
"""
from __future__ import annotations

import base64
import os
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import uvicorn

from alpa.content.provider import HTTPChatProvider, REGISTRY
from tests.mock_llm import EXPECTED_KEY, create_mock_llm


def make_pdf(lines):
    content = "BT /F1 18 Tf 72 700 Td " + \
        " ".join(f"({ln}) Tj 0 -24 Td" for ln in lines) + " ET"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
        + content.encode() + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offs = []
    for i, o in enumerate(objs, start=1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n".encode() + b"0000000000 65535 f \n"
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    out += (f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF").encode()
    return out


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.02)
    return server, th


def main():
    # 1) mock LLM endpoint
    llm_port = free_port()
    llm_server, _ = serve(create_mock_llm(), llm_port)
    REGISTRY.register(HTTPChatProvider(f"http://127.0.0.1:{llm_port}/v1",
                                       EXPECTED_KEY, "mock-model",
                                       name="http:mock-demo"))

    # 2) platform API
    db_path = Path(__file__).resolve().parents[1] / "demo.db"
    db_path.unlink(missing_ok=True)
    os.environ["ALPA_DB"] = f"sqlite:///{db_path}"
    os.environ["ALPA_LLM_PROVIDER"] = "http:mock-demo"
    from alpa.api.app import create_app
    api_port = free_port()
    api_server, _ = serve(create_app(os.environ["ALPA_DB"]), api_port)
    b = f"http://127.0.0.1:{api_port}"
    print(f"[demo] mock LLM on :{llm_port} · platform API on :{api_port}\n")

    # 3) PDF -> learning graph
    pdf = make_pdf(["Torque", "Power", "Friction"])
    r = httpx.post(b + "/api/ingest", json={
        "b64": base64.b64encode(pdf).decode(), "filename": "lecture.pdf",
        "provider": "http:mock-demo"}).json()
    print(f"[ingest] PDF -> {len(r['concepts'])} concepts, "
          f"{len(r['items_added'])} items {[i['item_type'] for i in r['items_added']]}, "
          f"{r['edges_added']} LLM-proposed edge(s), "
          f"{len(r['questions_rejected'])} rejected")

    # 4) approval gates
    g0 = httpx.get(b + "/api/graph").json()
    llm_edge = next(e for e in g0["edges"] if e["provenance"].startswith("llm:"))
    print(f"[graph] LLM edge {llm_edge['src']}->{llm_edge['dst']} traversable "
          f"BEFORE review: {llm_edge['traversable']}")
    for c in r["concepts"]:
        httpx.post(f"{b}/api/concepts/{c['id']}/approve")
    httpx.post(f"{b}/api/edges/{llm_edge['id']}/approve")
    g1 = httpx.get(b + "/api/graph").json()
    print(f"[graph] same edge traversable AFTER approval: "
          f"{next(e['traversable'] for e in g1['edges'] if e['id']==llm_edge['id'])}")

    # 5) adaptive session with provider-generated teaching content
    uid = httpx.post(b + "/api/users", json={"name": "demo"}).json()["id"]
    sid = httpx.post(b + f"/api/users/{uid}/sessions").json()["id"]
    teach = attempts = 0
    last_iv = None
    for _ in range(40):
        d = httpx.get(b + f"/api/sessions/{sid}/next").json()
        if d["intervention"] == "SESSION_WRAP":
            sid = httpx.post(b + f"/api/users/{uid}/sessions").json()["id"]
            continue
        if d["intervention"] in ("EXPLAIN", "EXAMPLE"):
            teach += 1
            print(f"[teach] {d['intervention']} on '{d['concept']}' via "
                  f"{d['payload']['content_provider']}: "
                  f"{d['payload']['generated_content'][:64]}...")
            continue
        item = d["payload"].get("item")
        if not item:
            continue
        attempts += 1
        last_iv = d["intervention_id"]
        httpx.post(b + f"/api/sessions/{sid}/attempts", json={
            "intervention_id": d["intervention_id"], "item_id": item["item_id"],
            "choice_index": 0, "latency_ms": 5200})
        if attempts >= 12 and teach >= 2:
            break

    # 6) trace reconstruction
    tr = httpx.get(b + f"/api/interventions/{last_iv}/trace").json()
    print(f"[trace] decision #{last_iv}: policy='{tr['intervention']['rationale']}' "
          f"p_correct={tr['prediction']['p_correct']:.3f} "
          f"outcome_correct={tr['outcome']['correct']}")
    print("\n[demo] full NEXT-block pipeline: OK")
    llm_server.should_exit = api_server.should_exit = True
    db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
