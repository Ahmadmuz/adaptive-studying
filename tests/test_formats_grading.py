"""Real file formats (PDF/PPTX/DOCX/TXT) through the ingestion pipeline, and
robust deterministic grading of banked non-MCQ items via the attempt path."""
import base64
import io
import json
import os
import socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from alpa.content.ingest import extract_text
from tests.mock_llm import EXPECTED_KEY, create_mock_llm


# ------------------------------------------------------------------ files ---

def make_docx(paragraphs: list[str]) -> bytes:
    import docx
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def make_pptx(texts: list[str]) -> bytes:
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # blank
    for i, t in enumerate(texts):
        box = slide.shapes.add_textbox(Inches(1), Inches(1 + i), Inches(6), Inches(1))
        box.text_frame.text = t
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def make_pdf(lines: list[str]) -> bytes:
    """Minimal valid PDF with an uncompressed text stream (hand-built xref)."""
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
    offsets = []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF").encode()
    return out


def test_extract_text_real_formats():
    assert "Work and Energy" in extract_text(make_docx(["Work and Energy", "Power"]), "u.docx")
    assert "Work and Energy" in extract_text(make_pptx(["Work and Energy", "Power"]), "u.pptx")
    assert "Work and Energy" in extract_text(make_pdf(["Work and Energy", "Power"]), "u.pdf")
    assert "Work and Energy" in extract_text(b"Work and Energy\nPower", "u.txt")


def _client(tmp_path, name):
    os.environ["ALPA_DB"] = f"sqlite:///{tmp_path}/{name}.db"
    from alpa.api.app import create_app
    return TestClient(create_app(os.environ["ALPA_DB"]))


@pytest.mark.parametrize("maker, filename", [
    (make_docx, "notes.docx"), (make_pptx, "slides.pptx"), (make_pdf, "sheet.pdf")])
def test_file_ingestion_end_to_end(tmp_path, maker, filename):
    with _client(tmp_path, "fmt") as client:
        payload = maker(["Work and Energy", "Power"])
        r = client.post("/api/ingest", json={
            "b64": base64.b64encode(payload).decode(),
            "filename": filename, "provider": "template:v1"}).json()
        codes = {c["code"] for c in r["concepts"]}
        assert {"work_and_energy", "power"} <= codes, f"{filename}: concepts missing"
        assert r["items_added"], f"{filename}: validated MCQ items must be banked"
        assert all(c["human_reviewed"] is False for c in r["concepts"])


# ------------------------------------------------------- non-MCQ grading ----

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def mock_server():
    app = create_mock_llm()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    th.join(timeout=5)


def test_numerical_and_short_answer_grading_through_attempts(tmp_path, mock_server):
    from alpa.content.provider import REGISTRY, HTTPChatProvider
    provider = HTTPChatProvider(mock_server, EXPECTED_KEY, "mock-model",
                                name="http:mock-grade")
    REGISTRY.register(provider)
    with _client(tmp_path, "grade") as client:
        r = client.post("/api/ingest", json={
            "topics": ["Power", "Friction"], "provider": "http:mock-grade"}).json()
        by_type = {i["item_type"]: i for i in r["items_added"]}
        assert {"numerical", "short_answer"} <= set(by_type), \
            "closed-form generated items must be banked"
        for c in r["concepts"]:
            client.post(f"/api/concepts/{c['id']}/approve")

        user = client.post("/api/users", json={"name": "grader"}).json()
        session = client.post(f"/api/users/{user['id']}/sessions").json()

        # craft interventions pointing at each generated item, then grade
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from alpa.db.models import Intervention
        eng = create_engine(os.environ["ALPA_DB"],
                            connect_args={"check_same_thread": False})
        DB = sessionmaker(bind=eng, expire_on_commit=False)

        def attempt(item_id, body_extra):
            with DB() as db:
                iv = Intervention(session_id=session["id"], user_id=user["id"],
                                  item_id=item_id, action="Q_MED", exploration=False,
                                  rationale="test:grading", policy_version_id=1,
                                  ts=0.0)
                db.add(iv)
                db.commit()
                iv_id = iv.id
            return client.post(f"/api/sessions/{session['id']}/attempts", json={
                "intervention_id": iv_id, "item_id": item_id,
                "latency_ms": 3000, **body_extra})

        num = by_type["numerical"]["item_id"]
        assert attempt(num, {"response_text": "250"}).json()["correct"] is True
        assert attempt(num, {"response_text": "253.5"}).json()["correct"] is True  # tolerance 5
        assert attempt(num, {"response_text": "300"}).json()["correct"] is False
        assert attempt(num, {"response_text": "not a number"}).json()["correct"] is False
        assert attempt(num, {"choice_index": 0}).status_code == 400  # wrong channel

        sa = by_type["short_answer"]["item_id"]
        assert attempt(sa, {"response_text": "Relative   MOTION"}).json()["correct"] is True
        assert attempt(sa, {"response_text": "gravity"}).json()["correct"] is False
        assert attempt(sa, {}).status_code == 400
