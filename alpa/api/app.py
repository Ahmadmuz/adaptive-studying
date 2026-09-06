"""FastAPI surface. Thin by design: all logic lives in alpa.engine / alpa.core."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .. import engine
from ..db.models import Base, SessionRow, User
from ..seed import load as seed_mod
from .schemas import AttemptIn, SessionOut, UserIn, UserOut

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def create_app(db_url: str | None = None) -> FastAPI:
    url = db_url or os.environ.get("ALPA_DB", "sqlite:///./alpa.db")
    engine_ = create_engine(url, connect_args={"check_same_thread": False}
                            if url.startswith("sqlite") else {})
    SessionLocal = sessionmaker(bind=engine_, expire_on_commit=False)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        Base.metadata.create_all(engine_)
        with SessionLocal() as db:
            seed_mod.seed(db)
        from ..content.provider import register_from_env
        _app.state.live_provider = register_from_env()
        yield

    app = FastAPI(title="Adaptive Learning Platform (Phase 2)", version="0.2.0",
                  lifespan=lifespan)
    app.state.db_factory = SessionLocal

    def _db():
        return SessionLocal()

    def _get_user(db, user_id: int) -> User:
        user = db.get(User, user_id)
        if user is None:
            raise HTTPException(404, "unknown user")
        return user

    def _get_session(db, session_id: int) -> SessionRow:
        s = db.get(SessionRow, session_id)
        if s is None:
            raise HTTPException(404, "unknown session")
        return s

    @app.get("/api/health")
    def health():
        from .. import engine as engine_mod
        live = getattr(app.state, "live_provider", None)
        return {"ok": True, "phase": 2, "llm_coupled": live is not None,
                "content_provider": engine_mod.CONTENT_PROVIDER}

    @app.post("/api/users", response_model=UserOut)
    def create_user(body: UserIn):
        with _db() as db:
            user = User(name=body.name, created_ts=time.time())
            db.add(user)
            db.commit()
            return {"id": user.id, "name": user.name}

    @app.post("/api/users/{user_id}/sessions", response_model=SessionOut)
    def start_session(user_id: int):
        with _db() as db:
            user = _get_user(db, user_id)
            s = SessionRow(user_id=user.id, started_ts=time.time())
            db.add(s)
            db.commit()
            return {"id": s.id, "user_id": user.id}

    @app.get("/api/sessions/{session_id}/next")
    def next_activity(session_id: int):
        with _db() as db:
            session = _get_session(db, session_id)
            user = _get_user(db, session.user_id)
            return engine.next_decision(db, user, session)

    @app.post("/api/sessions/{session_id}/attempts")
    def submit(session_id: int, body: AttemptIn):
        with _db() as db:
            session = _get_session(db, session_id)
            user = _get_user(db, session.user_id)
            try:
                return engine.submit_attempt(
                    db, user, session, body.intervention_id, body.item_id,
                    choice_index=body.choice_index, response_text=body.response_text,
                    latency_ms=body.latency_ms,
                    hint_level=body.hint_level, paste_detected=body.paste_detected,
                    confidence=body.confidence)
            except ValueError as e:
                raise HTTPException(400, str(e))

    @app.get("/api/users/{user_id}/progress")
    def get_progress(user_id: int):
        with _db() as db:
            user = _get_user(db, user_id)
            return engine.progress(db, user)

    @app.get("/api/users/{user_id}/state")
    def get_state(user_id: int):
        """Serializable student state (schema-versioned JSON)."""
        with _db() as db:
            user = _get_user(db, user_id)
            st = engine.student_state(db, user)
            return {"schema_version": st.schema_version, "user_id": st.user_id,
                    "graph_version": st.graph_version,
                    "fingerprint": st.fingerprint(), "state_json": st.to_json()}

    @app.get("/api/graph")
    def get_graph(include_unreviewed_llm: bool = False):
        """Concept graph with typed, provenance-carrying edges. Traversal used by
        the adaptive engine excludes unreviewed LLM edges by default."""
        with _db() as db:
            g = engine.build_graph(db)
            trav = {(e.src, e.dst, e.kind) for e in
                    g.traversable(include_unreviewed_llm=include_unreviewed_llm)}
            return {
                "graph_version": g.graph_version,
                "nodes": [{"id": n.id, "code": n.code, "title": n.title,
                           "description": n.description, "difficulty": n.difficulty}
                          for n in g.nodes.values()],
                "edges": [{"id": e.id, "src": e.src, "dst": e.dst, "kind": e.kind,
                           "confidence": e.confidence, "provenance": e.provenance,
                           "human_reviewed": e.human_reviewed,
                           "traversable": (e.src, e.dst, e.kind) in trav}
                          for e in g.edges],
                "topo_order": g.topo_order(),
            }

    # ----------------------------------------------------- content pipeline
    @app.get("/api/content/providers")
    def providers():
        from ..content.provider import REGISTRY
        return {"providers": REGISTRY.names()}

    @app.post("/api/ingest")
    def ingest(body: dict):
        """Ingest text or a manual topic list through the provider pipeline.
        Body: {text: str | filename+b64: str | topics: [str], provider: str}.
        Concept proposals enter UNREVIEWED; validated MCQ items join the bank
        attached to their concept (teachable only after approval)."""
        import base64
        from .. import engine as engine_mod
        from ..content.ingest import extract_text, ingest_text
        from ..content.provider import REGISTRY
        provider = REGISTRY.get(body.get("provider", engine_mod.CONTENT_PROVIDER))
        if "topics" in body:
            text = "\n".join(t.strip() for t in body["topics"] if str(t).strip())
            source = "manual_topics"
        elif "b64" in body and "filename" in body:
            text = extract_text(base64.b64decode(body["b64"]), body["filename"])
            source = body["filename"]
        elif "text" in body:
            text = body["text"]
            source = body.get("source", "inline")
        else:
            raise HTTPException(400, "provide text, topics, or b64+filename")
        report = ingest_text(text, provider, source=source)
        with _db() as db:
            persisted = engine.ingest_report_to_db(db, report)
        return {
            "provider": report.provider, "source": source,
            "n_segments": report.n_segments,
            "concepts": persisted["concepts_added"],
            "items_added": persisted["items_added"],
            "edges_added": persisted["edges_added"],
            "items_deferred": persisted["items_deferred"],
            "questions_rejected": persisted["questions_rejected"],
        }

    @app.post("/api/concepts/{concept_id}/approve")
    def approve(concept_id: int):
        """Human review gate: approved concepts become part of the adaptive
        loop; until then they are inert proposals."""
        with _db() as db:
            try:
                cid = engine.approve_concept(db, concept_id)
            except ValueError as e:
                raise HTTPException(404, str(e))
            return {"concept_id": cid, "human_reviewed": True}

    @app.post("/api/edges/{relationship_id}/approve")
    def approve_edge(relationship_id: int):
        """Human review gate for LLM-proposed relationships: approved edges
        become traversable by the adaptive engine."""
        with _db() as db:
            try:
                rid = engine.approve_edge(db, relationship_id)
            except ValueError as e:
                raise HTTPException(404, str(e))
            return {"relationship_id": rid, "human_reviewed": True}

    @app.get("/api/interventions/{intervention_id}/trace")
    def get_trace(intervention_id: int):
        with _db() as db:
            out = engine.trace(db, intervention_id)
            if out is None:
                raise HTTPException(404, "unknown intervention")
            return out

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()
