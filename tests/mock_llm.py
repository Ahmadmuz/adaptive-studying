"""Deterministic mock OpenAI-compatible server for provider validation.

Runs as an ASGI app behind httpx.ASGITransport: real request/response cycles
through the full HTTP provider code path (auth header, chat/completions shape,
response_format negotiation), without any external service. Responses are
scripted by prompt markers so tests are deterministic.
"""
from __future__ import annotations

import json
import re

from fastapi import FastAPI, Request

EXPECTED_KEY = "test-key-123"


def _route(system: str, user: str) -> str:
    if "assessment item" in system.lower() or "one assessment item" in system.lower():
        m = re.search(r'"concept":\s*"([^"]+)"', user)
        concept = m.group(1) if m else "Unknown"
        if concept == "Torque":
            return json.dumps({
                "question": "Torque is best described as:",
                "item_type": "multiple_choice",
                "choices": ["Force times lever arm", "Mass times velocity",
                            "Energy per unit time", "Momentum change"],
                "correct_index": 0,
                "answer": "Force times lever arm",
                "explanation": "Torque = r x F.",
                "difficulty": 0.6,
                "prerequisites": ["Force"]})
        if concept == "Power":
            return json.dumps({
                "question": "A motor does 500 J of work in 2 s. Power output in watts?",
                "item_type": "numerical",
                "answer": "250",
                "tolerance": 5,
                "explanation": "P = W / t = 500 / 2.",
                "difficulty": 0.5,
                "prerequisites": ["Work"]})
        if concept == "Friction":
            return json.dumps({
                "question": "In one short phrase: what does kinetic friction oppose?",
                "item_type": "short_answer",
                "answer": "relative motion",
                "explanation": "Kinetic friction opposes relative sliding.",
                "difficulty": 0.4})
        if concept == "BadConcept":
            return json.dumps({
                "question": "A flawed item follows:",
                "item_type": "multiple_choice",
                "choices": ["Same answer", "same  answer", "Other"],
                "correct_index": 0,
                "answer": "Same answer",
                "explanation": "duplicate choices",
                "difficulty": 0.5})
        if concept == "Garbled":
            return "Sure! Here is the item: ```json {question: oops ```"
        return json.dumps({
            "question": f"Which statement best describes {concept}?",
            "item_type": "multiple_choice",
            "choices": [f"Correct view of {concept}", "Wrong view", "Unrelated"],
            "correct_index": 0,
            "answer": f"Correct view of {concept}",
            "explanation": "mock", "difficulty": 0.5})
    if "concept proposals" in system:
        names = [ln.strip() for ln in user.splitlines()
                 if ln.strip() and not ln.startswith("Material")][:12]
        return json.dumps({"concepts": [
            {"name": n, "description": f"extracted: {n}", "confidence": 0.8}
            for n in names if 3 <= len(n) <= 60]})
    if "learning relationships" in system:
        m = re.search(r"Concepts:\s*(.+)", user)
        names = [n.strip() for n in m.group(1).split(",")] if m else []
        rels = []
        if len(names) >= 2:
            rels.append({"src": names[1], "dst": names[0],
                         "kind": "PREREQUISITE", "confidence": 0.9})
        return json.dumps({"relationships": rels})
    m = re.search(r"'([^']+)'", user)
    topic = m.group(1) if m else "the topic"
    return f"MOCK EXPLANATION about {topic} tailored to the student state."


def create_mock_llm() -> FastAPI:
    app = FastAPI()
    app.state.log = []

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        app.state.log.append({
            "auth": request.headers.get("authorization", ""),
            "model": body.get("model"),
            "response_format": body.get("response_format"),
            "system": body["messages"][0]["content"],
            "user": body["messages"][1]["content"],
        })
        if request.headers.get("authorization") != f"Bearer {EXPECTED_KEY}":
            return {"error": {"message": "invalid api key", "type": "auth"}}
        content = _route(app.state.log[-1]["system"], app.state.log[-1]["user"])
        return {"choices": [{"message": {"content": content}}]}

    return app
