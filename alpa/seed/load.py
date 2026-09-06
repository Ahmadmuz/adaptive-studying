"""Seed loader: course + graph v1 + items + model/policy version registration."""
from __future__ import annotations

import json
import time

from sqlalchemy.orm import Session

from ..core import policy as policy_mod
from ..core.types import GlobalParams
from ..db.models import (
    Concept, ConceptRelationship, Course, GraphVersion, Item, ItemConcept,
    ModelVersion, PolicyVersion,
)
from . import mechanics


MODEL_NAME = "hybrid-pfa-irt-v1"
POLICY_NAME = "rules-v1"


def is_seeded(db: Session) -> bool:
    return db.query(Course.id).filter_by(code=mechanics.COURSE["code"]).first() is not None


def seed(db: Session) -> None:
    if is_seeded(db):
        return
    now = time.time()
    gp = GlobalParams()

    course = Course(code=mechanics.COURSE["code"], title=mechanics.COURSE["title"],
                    description=mechanics.COURSE["description"])
    db.add(course)
    db.flush()

    gv = GraphVersion(course_id=course.id, label="v1-hand-authored", created_ts=now)
    db.add(gv)
    db.flush()

    code_to_id: dict[str, int] = {}
    for topo, (code, title, desc, _prereqs) in enumerate(mechanics.CONCEPTS):
        c = Concept(course_id=course.id, graph_version_id=gv.id, topo_index=topo,
                    code=code, title=title, description=desc,
                    mu=mechanics.GLOBAL_MU, sigma=mechanics.GLOBAL_SIGMA)
        db.add(c)
        db.flush()
        code_to_id[code] = c.id

    for topo, (code, _t, _d, prereqs) in enumerate(mechanics.CONCEPTS):
        for p in prereqs:
            db.add(ConceptRelationship(
                graph_version_id=gv.id, src_concept_id=code_to_id[p],
                dst_concept_id=code_to_id[code], edge_type="PREREQUISITE",
                confidence=1.0, provenance="human", human_reviewed=True))
    for edge_type, src, dst in mechanics.EXTRA_EDGES:
        db.add(ConceptRelationship(
            graph_version_id=gv.id, src_concept_id=code_to_id[src],
            dst_concept_id=code_to_id[dst], edge_type=edge_type,
            confidence=1.0, provenance="human", human_reviewed=True))

    for i, (code, kind, stem, choices, correct, b, a) in enumerate(mechanics.ITEMS):
        item = Item(course_id=course.id, graph_version_id=gv.id, code=f"{code}-q{i:03d}",
                    stem=stem, choices_json=json.dumps(choices), correct_index=correct,
                    b=b, a=a, kind=kind)
        db.add(item)
        db.flush()
        db.add(ItemConcept(item_id=item.id, concept_id=code_to_id[code], is_primary=True))

    mv = ModelVersion(name=MODEL_NAME, kind="global_model",
                      params_json=json.dumps(gp.__dict__), created_ts=now)
    db.add(mv)
    pv = PolicyVersion(
        name=POLICY_NAME,
        table_json=json.dumps({
            "review_threshold": policy_mod.REVIEW_THRESHOLD,
            "slip_latency_ms": policy_mod.SLIP_LATENCY_MS,
            "misconception_streak": policy_mod.MISCONCEPTION_STREAK,
            "prereq_weak_p": policy_mod.PREREQ_WEAK_P,
            "target_p": {"easy": policy_mod.TARGET_P_EASY, "medium": policy_mod.TARGET_P_MED,
                         "hard": policy_mod.TARGET_P_HARD},
            "fatigue_max_attempts": policy_mod.FATIGUE_MAX_ATTEMPTS,
            "epsilon": policy_mod.EXPLORATION_EPSILON,
            "info_gain": {"p_lo": policy_mod.INFO_GAIN_P_LO,
                          "p_hi": policy_mod.INFO_GAIN_P_HI,
                          "var_min": policy_mod.INFO_GAIN_VAR_MIN},
        }),
        created_ts=now)
    db.add(pv)
    db.commit()
