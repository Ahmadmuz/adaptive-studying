"""Orchestration layer: DB <-> core math.

Responsibilities:
- lazy per-concept state creation from population priors (cold start),
- diagnostic phase, then the adaptive policy,
- prediction/intervention logging (the audit + future bandit dataset),
- attempt grading, state transition, mastery hysteresis,
- progress and decision-trace read models.

Global parameters are read from `model_versions` — the online path never writes
them (Phase 1 D8).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import asdict
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.orm import Session

from .core import diagnostic, model, policy
from .core.graph import EDGE_KINDS, ConceptGraph, ConceptNode, Edge
from .core.state import StudentState
from .core.types import (
    AttemptRecord, ConceptSnapshot, Decision, GlobalParams, ItemSpec, QUESTION_ACTIONS,
)
from .db.models import (
    Attempt, Concept, ConceptRelationship, Course, GraphVersion, Intervention,
    Item, ItemConcept, ModelPrediction, ModelVersion, PolicyVersion, SessionRow,
    StudentConceptState, User,
)

DIAGNOSTIC_BUDGET = 8
CONTENT_PROVIDER = os.environ.get("ALPA_LLM_PROVIDER", "template:v1")
TEACH_ACTIONS = ("EXPLAIN", "EXAMPLE", "SIMPLIFY", "BREAK_TASK")


# ---------------------------------------------------------------- helpers -----

def get_global_params(db: Session) -> GlobalParams:
    mv = db.query(ModelVersion).order_by(ModelVersion.id.desc()).first()
    return GlobalParams(**json.loads(mv.params_json))


def _model_version_id(db: Session) -> int:
    return db.query(ModelVersion.id).order_by(ModelVersion.id.desc()).first()[0]


def _policy_version_id(db: Session) -> int:
    return db.query(PolicyVersion.id).order_by(PolicyVersion.id.desc()).first()[0]


def _course_ctx(db: Session):
    course = db.query(Course).first()
    gv = db.query(GraphVersion).filter_by(course_id=course.id).order_by(GraphVersion.id.desc()).first()
    return course, gv


def _snap_from_row(row: StudentConceptState, code: str) -> ConceptSnapshot:
    return ConceptSnapshot(
        concept_id=row.concept_id, code=code, theta=row.theta, var=row.var,
        succ=row.succ, fail=row.fail, stability=row.stability,
        last_success_ts=row.last_success_ts, last_update_ts=row.last_update_ts,
        mastered=row.mastered, exposure=row.exposure,
    )


def _write_row(row: StudentConceptState, snap: ConceptSnapshot) -> None:
    row.theta, row.var = snap.theta, snap.var
    row.succ, row.fail = snap.succ, snap.fail
    row.stability = snap.stability
    row.last_success_ts = snap.last_success_ts
    row.last_update_ts = snap.last_update_ts
    row.mastered = snap.mastered
    row.exposure = snap.exposure


def ensure_states(db: Session, user: User, gv: GraphVersion) -> dict[int, StudentConceptState]:
    rows = {r.concept_id: r for r in db.query(StudentConceptState)
            .filter_by(user_id=user.id, graph_version_id=gv.id)}
    for c in db.query(Concept).filter_by(graph_version_id=gv.id, human_reviewed=True):
        if c.id not in rows:
            row = StudentConceptState(user_id=user.id, graph_version_id=gv.id, concept_id=c.id,
                                      theta=c.mu, var=c.sigma ** 2)
            db.add(row)
            db.flush()
            rows[c.id] = row
    return rows


def build_graph(db: Session, gv: GraphVersion | None = None) -> ConceptGraph:
    """Provenance-aware concept graph straight from the versioned DB tables."""
    if gv is None:
        _course, gv = _course_ctx(db)
    g = ConceptGraph(graph_version=gv.id)
    for c in db.query(Concept).filter_by(graph_version_id=gv.id):
        g.add_node(ConceptNode(id=c.id, code=c.code, title=c.title,
                               description=c.description, difficulty=c.mu))
    for rel in db.query(ConceptRelationship).filter_by(graph_version_id=gv.id):
        if rel.edge_type not in EDGE_KINDS:
            continue  # unknown future edge kinds must not break traversal
        g.add_edge(Edge(src=rel.src_concept_id, dst=rel.dst_concept_id,
                        kind=rel.edge_type, confidence=rel.confidence,
                        provenance=rel.provenance, human_reviewed=rel.human_reviewed,
                        id=rel.id))
    return g


def student_state(db: Session, user: User) -> StudentState:
    """Serializable, versioned snapshot of a learner's full concept state."""
    _course, gv = _course_ctx(db)
    rows = ensure_states(db, user, gv)
    concepts = {c.id: c for c in db.query(Concept).filter_by(graph_version_id=gv.id, human_reviewed=True)}
    st = StudentState(user_id=user.id, graph_version=gv.id)
    for cid, row in rows.items():
        st.concepts[cid] = _snap_from_row(row, concepts[cid].code)
    return st


def _itemspecs(db: Session, course: Course, gv: GraphVersion) -> tuple[list[ItemSpec], dict[int, int]]:
    """All items + map item_id -> primary concept id. Items attached to
    unreviewed concepts are never served — the approval gate covers items too."""
    reviewed = {c.id for c in db.query(Concept.id)
                .filter_by(graph_version_id=gv.id, human_reviewed=True)}
    primary = {ic.item_id: ic.concept_id for ic in db.query(ItemConcept).filter_by(is_primary=True)}
    specs = []
    for it in db.query(Item).filter_by(course_id=course.id, graph_version_id=gv.id):
        cid = primary.get(it.id)
        if cid is None or cid not in reviewed:
            continue
        specs.append(ItemSpec(item_id=it.id, stem=it.stem,
                              choices=tuple(json.loads(it.choices_json)),
                              correct_index=it.correct_index, concept_ids=(cid,),
                              b=it.b, a=it.a, kind=it.kind,
                              item_type=it.item_type or "multiple_choice"))
    return specs, primary


# ------------------------------------------------------------- decisions ------

def next_decision(db: Session, user: User, session: SessionRow, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    course, gv = _course_ctx(db)
    gp = get_global_params(db)
    state_rows = ensure_states(db, user, gv)
    concepts = {c.id: c for c in db.query(Concept).filter_by(graph_version_id=gv.id, human_reviewed=True)}
    snapshots = {cid: _snap_from_row(row, concepts[cid].code) for cid, row in state_rows.items()}
    topo_order = [c.id for c in sorted(concepts.values(), key=lambda c: c.topo_index)]
    graph = build_graph(db, gv)
    prereqs = {cid: graph.prereqs(cid) for cid in concepts}
    items, _primary = _itemspecs(db, course, gv)
    items_by_concept: dict[int, list[ItemSpec]] = {}
    for it in items:
        items_by_concept.setdefault(it.concept_ids[0], []).append(it)
    attempted_ids = {a.item_id for a in db.query(Attempt.item_id).filter_by(user_id=user.id)}

    total_attempts = db.query(func.count(Attempt.id)).filter_by(user_id=user.id).scalar()
    session_attempts = db.query(func.count(Attempt.id)).filter_by(session_id=session.id).scalar()

    features: dict = {}
    if diagnostic.diagnostic_budget(total_attempts, DIAGNOSTIC_BUDGET):
        picked = diagnostic.pick_diagnostic_item(
            snapshots, items, attempted_ids,
            fresh_factory=lambda cid: snapshots[cid], now=now, gp=gp)
        if picked is None:
            decision = Decision("SESSION_WRAP", None, None, None, "diagnostic:item_pool_exhausted")
        else:
            item, p = picked
            feats = model.predict(snapshots[item.concept_ids[0]], item, now, gp)
            decision = Decision("DIAGNOSTIC", item.concept_ids[0], item.item_id, p,
                                "diagnostic:max_fisher_information", payload={"fisher_p": p})
            features = feats
    else:
        recent_attempts: dict[int, list[AttemptRecord]] = {}
        rows = (db.query(Attempt, ItemConcept.concept_id)
                .join(ItemConcept, (ItemConcept.item_id == Attempt.item_id) & ItemConcept.is_primary)
                .filter(Attempt.user_id == user.id)
                .order_by(Attempt.ts.desc()).limit(40).all())
        for att, cid in reversed(rows):
            recent_attempts.setdefault(cid, []).append(
                AttemptRecord(correct=att.correct, latency_ms=att.latency_ms,
                              ts=att.ts, paste_detected=att.paste_detected))
        for lst in recent_attempts.values():
            lst.sort(key=lambda a: a.ts)

        content_shown: dict[int, int] = dict(
            db.query(Intervention.concept_id, func.count(Intervention.id))
            .filter(Intervention.user_id == user.id,
                    Intervention.action.in_(("EXPLAIN", "EXAMPLE")),
                    Intervention.concept_id.is_not(None))
            .group_by(Intervention.concept_id).all())

        # remediation delivered since the concept's last attempt (anti-loop)
        last_attempt_ts = dict(
            db.query(ItemConcept.concept_id, func.max(Attempt.ts))
            .join(Attempt, Attempt.item_id == ItemConcept.item_id)
            .filter(Attempt.user_id == user.id, ItemConcept.is_primary == True)  # noqa: E712
            .group_by(ItemConcept.concept_id).all())
        remediated: dict[int, int] = {}
        for cid, ts in db.query(Intervention.concept_id, Intervention.ts).filter(
                Intervention.user_id == user.id,
                Intervention.action.in_(("SIMPLIFY", "BREAK_TASK",
                                         "REVISIT_PREREQUISITE")),
                Intervention.concept_id.is_not(None)):
            if ts > last_attempt_ts.get(cid, float("-inf")):
                remediated[cid] = remediated.get(cid, 0) + 1

        ctx = policy.PolicyContext(
            snapshots=snapshots, topo_order=topo_order, prereqs=prereqs,
            items_by_concept=items_by_concept, attempted_item_ids=attempted_ids,
            recent_attempts=recent_attempts, content_shown=content_shown,
            remediated=remediated,
            now=now, session_attempts=session_attempts, gp=gp, rng=random.Random())
        decision = policy.decide(ctx)
        if decision.item_id is not None:
            item = next(it for it in items if it.item_id == decision.item_id)
            features = model.predict(snapshots[item.concept_ids[0]], item, now, gp)

    intervention = Intervention(
        session_id=session.id, user_id=user.id, concept_id=decision.concept_id,
        item_id=decision.item_id, action=decision.intervention,
        exploration=decision.exploration, rationale=decision.rationale,
        policy_version_id=_policy_version_id(db), ts=now)
    db.add(intervention)
    db.flush()

    # Content generation hook: the ML engine decides WHAT/WHY (decision + spec),
    # the provider decides HOW. Spec and content are part of the logged decision.
    if decision.intervention in TEACH_ACTIONS and decision.concept_id is not None:
        from .content.provider import REGISTRY, GenerationSpec
        snap_c = snapshots[decision.concept_id]
        spec = GenerationSpec(
            concept=concepts[decision.concept_id].code,
            mastery=round(model.mastery_prob(snap_c, now, gp), 4),
            uncertainty=round(model.eff_var(snap_c, now, gp) ** 0.5, 4),
            difficulty=round(decision.target_p if decision.target_p is not None else 0.55, 2),
            intervention=decision.intervention,
            prerequisites=tuple(concepts[p].code
                                for p in prereqs.get(decision.concept_id, [])))
        try:
            provider = REGISTRY.get(CONTENT_PROVIDER)
            content = (provider.generate_explanation(spec)
                       if decision.intervention == "EXPLAIN"
                       else provider.generate_example(spec))
            decision.payload["content_spec"] = {"concept": spec.concept,
                                                "intervention": spec.intervention}
            decision.payload["generation_spec"] = asdict(spec)
            decision.payload["generated_content"] = content
            decision.payload["content_provider"] = provider.name
        except KeyError:
            pass  # no provider configured: payload keeps the Phase 5 stub contract

    db.add(ModelPrediction(
        intervention_id=intervention.id, model_version_id=_model_version_id(db),
        features_json=json.dumps({k: v for k, v in features.items()}, default=float),
        p_correct=features.get("p_correct") if features else None,
        uncertainty=features.get("var_eff") ** 0.5 if features and "var_eff" in features else None,
        decision_json=json.dumps(asdict(decision), default=str), ts=now))
    db.commit()

    payload = dict(decision.payload)
    if decision.item_id is not None:
        it = next(i for i in items if i.item_id == decision.item_id)
        payload["item"] = {"item_id": it.item_id, "stem": it.stem,
                           "choices": list(it.choices), "kind": it.kind,
                           "item_type": it.item_type}
    concept_title = concepts[decision.concept_id].title if decision.concept_id else None
    return {
        "intervention_id": intervention.id,
        "intervention": decision.intervention,
        "concept_id": decision.concept_id,
        "concept": concept_title,
        "target_p": decision.target_p,
        "p_correct": features.get("p_correct") if features else None,
        "uncertainty": (features.get("var_eff") ** 0.5) if features and "var_eff" in features else None,
        "rationale": decision.rationale,
        "exploration": decision.exploration,
        "payload": payload,
    }


# -------------------------------------------------------------- attempts ------

def submit_attempt(db: Session, user: User, session: SessionRow, intervention_id: int,
                   item_id: int, choice_index: int | None = None,
                   response_text: str | None = None, latency_ms: float | None = None,
                   hint_level: int = 0, paste_detected: bool = False,
                   confidence: float | None = None, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    intervention = db.get(Intervention, intervention_id)
    if intervention is None or intervention.session_id != session.id:
        raise ValueError("unknown intervention for this session")
    if intervention.item_id is not None and intervention.item_id != item_id:
        raise ValueError("attempted item does not match the served decision")
    item_row = db.get(Item, item_id)
    if item_row is None:
        raise ValueError("unknown item")
    gp = get_global_params(db)
    course, gv = _course_ctx(db)
    state_rows = ensure_states(db, user, gv)
    concepts = {c.id: c for c in db.query(Concept).filter_by(graph_version_id=gv.id, human_reviewed=True)}
    primary_cid = (db.query(ItemConcept.concept_id)
                   .filter_by(item_id=item_id, is_primary=True).first())[0]

    item = ItemSpec(item_id=item_row.id, stem=item_row.stem,
                    choices=tuple(json.loads(item_row.choices_json)),
                    correct_index=item_row.correct_index, concept_ids=(primary_cid,),
                    b=item_row.b, a=item_row.a, kind=item_row.kind)

    # ---- grading dispatch: MCQ by choice; closed forms by deterministic grader.
    # External/rubric grading is rejected here: it never enters the core model.
    item_type = item_row.item_type or "multiple_choice"
    if item_type == "multiple_choice":
        if choice_index is None:
            raise ValueError("multiple_choice items require choice_index")
        correct = choice_index == item_row.correct_index
        grading_method = "exact_choice"
    else:
        from .content.questions import QuestionSpec
        if response_text is None:
            raise ValueError(f"{item_type} items require response_text")
        gspec = QuestionSpec(
            question=item_row.stem, answer=item_row.answer_text or "",
            explanation=item_row.explanation or "", concept="_",
            difficulty=0.5, item_type=item_type,
            grading_method="numeric_tolerance" if item_type == "numerical"
            else "normalized_match",
            tolerance=item_row.tolerance, rubric=item_row.rubric)
        graded, grading_method = gspec.grade(response_text)
        if graded is None:
            raise ValueError("item requires an external grader; not admitted "
                             "to the core student model")
        correct = bool(graded)
    y = 1 if correct else 0

    row = state_rows[primary_cid]
    snap_before = _snap_from_row(row, concepts[primary_cid].code)
    p_before = model.mastery_prob(snap_before, now, gp)
    snap_after = model.update(snap_before, item, y, now, gp)
    snap_after.mastered = model.apply_mastery_hysteresis(snap_after, now, gp)
    _write_row(row, snap_after)

    attempt_no = (db.query(func.count(Attempt.id))
                  .filter_by(user_id=user.id, item_id=item_id).scalar() + 1)
    db.add(Attempt(session_id=session.id, user_id=user.id, item_id=item_id,
                   intervention_id=intervention.id, choice_index=choice_index,
                   correct=correct, latency_ms=latency_ms, attempt_no=attempt_no,
                   hint_level_reached=hint_level, paste_detected=paste_detected,
                   confidence=confidence,
                   error_tag="possible_gaming" if paste_detected else None,
                   response_text=response_text, grading_method=grading_method, ts=now))
    db.commit()

    return {
        "correct": correct,
        "correct_index": item_row.correct_index,
        "concept": concepts[primary_cid].title,
        "mastery_before": {"mastered": snap_before.mastered, "p_mastery": p_before},
        "mastery_after": {"mastered": snap_after.mastered,
                          "p_mastery": model.mastery_prob(snap_after, now, gp)},
        "theta": snap_after.theta,
        "succ": snap_after.succ,
        "fail": snap_after.fail,
    }


# -------------------------------------------------------------- read side -----

# ------------------------------------------------------------- ingestion ----

def ingest_report_to_db(db: Session, report) -> dict:
    """Persist an IngestReport: concept proposals enter UNREVIEWED (they cannot
    steer adaptation until approved); accepted multiple-choice questions become
    bank items with provenance. Non-MCQ item types are validated and graded at
    the QuestionSpec level but are not admitted to the attempt pipeline yet —
    logged as deferred, not silently dropped."""
    course, gv = _course_ctx(db)
    added_concepts, added_items, deferred = [], [], []
    concepts = {c.code: c for c in db.query(Concept).filter_by(graph_version_id=gv.id)}
    next_topo = max((c.topo_index for c in concepts.values()), default=-1) + 1
    for prop in report.concept_proposals:
        code = re.sub(r"\W+", "_", prop["name"])[:60].lower() or f"concept_{len(concepts)}"
        base, k = code, 2
        while code in concepts:
            code, k = f"{base}_{k}", k + 1
        c = Concept(course_id=course.id, graph_version_id=gv.id, topo_index=next_topo,
                    code=code, title=prop["name"], description=prop.get("description", ""),
                    mu=-0.5, sigma=1.0, human_reviewed=False, provenance=prop["provenance"])
        db.add(c)
        db.flush()
        concepts[code] = c
        next_topo += 1
        added_concepts.append({"id": c.id, "code": code, "title": c.title,
                               "human_reviewed": False, "provenance": c.provenance})
    title_to_code = {}
    for code, c in concepts.items():
        title_to_code[code.lower()] = code
        title_to_code[c.title.lower()] = code
    for q in report.questions_accepted:
        code = title_to_code.get(q.concept.lower())
        if code is None:
            report.questions_rejected.append(
                {"question": q.question[:80], "errors": [f"unknown concept {q.concept!r}"]})
            continue
        if q.item_type not in ("multiple_choice", "numerical", "short_answer"):
            deferred.append({"question": q.question[:80], "item_type": q.item_type,
                             "reason": "requires external rubric grading; kept out of "
                                       "the core model"})
            continue
        p = min(max(q.difficulty, 0.01), 0.99)
        item = Item(course_id=course.id, graph_version_id=gv.id,
                    code=f"gen:{report.provider}:{uuid4().hex[:8]}",
                    stem=q.question,
                    choices_json=json.dumps(q.choices if q.item_type == "multiple_choice" else []),
                    correct_index=q.correct_index if q.item_type == "multiple_choice" else 0,
                    b=-math.log(p / (1 - p)), a=1.0, kind="practice",
                    provenance=q.provenance, item_type=q.item_type,
                    answer_text=q.answer, tolerance=q.tolerance,
                    explanation=q.explanation or None)
        db.add(item)
        db.flush()
        db.add(ItemConcept(item_id=item.id, concept_id=concepts[code].id, is_primary=True))
        added_items.append({"item_id": item.id, "concept": code, "item_type": q.item_type})

    # LLM-proposed relationships: persisted with provenance, UNREVIEWED, and
    # excluded from adaptive traversal until human approval (graph module gate).
    edges_added = 0
    for rel in report.relationship_proposals:
        src = title_to_code.get(str(rel.get("src", "")).lower())
        dst = title_to_code.get(str(rel.get("dst", "")).lower())
        kind = rel.get("kind", "RELATED")
        if src is None or dst is None or src == dst or kind not in EDGE_KINDS:
            continue
        try:
            conf = float(rel.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        db.add(ConceptRelationship(
            graph_version_id=gv.id, src_concept_id=concepts[src].id,
            dst_concept_id=concepts[dst].id, edge_type=kind,
            confidence=min(max(conf, 0.0), 1.0),
            provenance=rel.get("provenance", f"llm:{report.provider}"),
            human_reviewed=False))
        edges_added += 1
    db.commit()
    return {"concepts_added": added_concepts, "items_added": added_items,
            "edges_added": edges_added, "items_deferred": deferred,
            "questions_rejected": report.questions_rejected}


def approve_concept(db: Session, concept_id: int) -> int:
    """Human review gate: only approved concepts become teachable/testable."""
    c = db.get(Concept, concept_id)
    if c is None:
        raise ValueError("unknown concept")
    c.human_reviewed = True
    db.commit()
    return c.id


def approve_edge(db: Session, relationship_id: int) -> int:
    """Human review gate for LLM-proposed relationships: only approved edges
    become traversable by the adaptive engine."""
    r = db.get(ConceptRelationship, relationship_id)
    if r is None:
        raise ValueError("unknown relationship")
    r.human_reviewed = True
    db.commit()
    return r.id


def progress(db: Session, user: User, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    course, gv = _course_ctx(db)
    gp = get_global_params(db)
    state_rows = ensure_states(db, user, gv)
    concepts = sorted(db.query(Concept).filter_by(graph_version_id=gv.id, human_reviewed=True),
                      key=lambda c: c.topo_index)
    out = []
    mastered = 0
    for c in concepts:
        snap = _snap_from_row(state_rows[c.id], c.code)
        p_m = model.mastery_prob(snap, now, gp)
        r = model.retention(snap, now, gp)
        due = model.review_due_days(snap, now, gp)
        mastered += int(snap.mastered)
        out.append({"code": c.code, "title": c.title, "mastered": snap.mastered,
                    "p_mastery": round(p_m, 4), "theta": round(snap.theta, 3),
                    "retention": round(r, 3), "review_due_days": None if due is None else round(due, 2),
                    "succ": snap.succ, "fail": snap.fail, "exposure": snap.exposure})
    total_attempts = db.query(func.count(Attempt.id)).filter_by(user_id=user.id).scalar()
    return {"course": course.title, "mastered": mastered, "total_concepts": len(concepts),
            "total_attempts": total_attempts, "concepts": out}


def trace(db: Session, intervention_id: int) -> dict | None:
    """Full traceability for one decision: policy record + model prediction + outcome."""
    iv = db.get(Intervention, intervention_id)
    if iv is None:
        return None
    pred = db.query(ModelPrediction).filter_by(intervention_id=intervention_id).first()
    att = db.query(Attempt).filter_by(intervention_id=intervention_id).first()
    return {
        "intervention": {
            "id": iv.id, "action": iv.action, "concept_id": iv.concept_id,
            "item_id": iv.item_id, "exploration": iv.exploration,
            "rationale": iv.rationale, "policy_version_id": iv.policy_version_id, "ts": iv.ts},
        "prediction": None if pred is None else {
            "id": pred.id, "model_version_id": pred.model_version_id,
            "features": json.loads(pred.features_json), "p_correct": pred.p_correct,
            "uncertainty": pred.uncertainty, "decision": json.loads(pred.decision_json),
            "ts": pred.ts},
        "outcome": None if att is None else {
            "id": att.id, "item_id": att.item_id, "choice_index": att.choice_index,
            "correct": att.correct, "latency_ms": att.latency_ms,
            "hint_level_reached": att.hint_level_reached, "ts": att.ts},
    }
