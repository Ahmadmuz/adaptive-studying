"""SQLAlchemy 2.0 schema. Design invariants from Phase 1 §4:

- the concept graph is versioned; student states reference the graph version they
  were computed under;
- `model_predictions` is append-only audit log (prediction <-> intervention <->
  attempt join chain);
- `interventions` record the exploration flag and rule-cell rationale;
- global model parameters live in `model_versions` (refit offline, release-gated);
  the online path never writes them.
Timestamps are epoch seconds (Float) — arithmetic-first for the math layer.
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    created_ts: Mapped[float] = mapped_column(Float)


class Course(Base):
    __tablename__ = "courses"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(40), unique=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")


class GraphVersion(Base):
    __tablename__ = "concept_graph_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"))
    label: Mapped[str] = mapped_column(String(40))
    created_ts: Mapped[float] = mapped_column(Float)


class Concept(Base):
    __tablename__ = "concepts"
    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"))
    graph_version_id: Mapped[int] = mapped_column(ForeignKey("concept_graph_versions.id"))
    topo_index: Mapped[int] = mapped_column(Integer)          # prerequisite-respecting order
    code: Mapped[str] = mapped_column(String(60))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    mu: Mapped[float] = mapped_column(Float, default=-0.5)    # population prior mean
    sigma: Mapped[float] = mapped_column(Float, default=1.0)  # population prior sd
    theta_mastery: Mapped[float | None] = mapped_column(Float, nullable=True)  # per-concept override
    human_reviewed: Mapped[bool] = mapped_column(Boolean, default=True)
    provenance: Mapped[str] = mapped_column(String(200), default="human")


class ConceptRelationship(Base):
    __tablename__ = "concept_relationships"
    id: Mapped[int] = mapped_column(primary_key=True)
    graph_version_id: Mapped[int] = mapped_column(ForeignKey("concept_graph_versions.id"))
    src_concept_id: Mapped[int] = mapped_column(ForeignKey("concepts.id"))  # prerequisite
    dst_concept_id: Mapped[int] = mapped_column(ForeignKey("concepts.id"))  # depends on src
    edge_type: Mapped[str] = mapped_column(String(30), default="PREREQUISITE")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    provenance: Mapped[str] = mapped_column(String(200), default="human")   # human | llm:<model>
    human_reviewed: Mapped[bool] = mapped_column(Boolean, default=True)


class Item(Base):
    __tablename__ = "items"
    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"))
    graph_version_id: Mapped[int] = mapped_column(ForeignKey("concept_graph_versions.id"))
    code: Mapped[str] = mapped_column(String(60))
    stem: Mapped[str] = mapped_column(Text)
    choices_json: Mapped[str] = mapped_column(Text)           # JSON list of strings
    correct_index: Mapped[int] = mapped_column(Integer)
    b: Mapped[float] = mapped_column(Float)                   # difficulty prior; becomes empirical
    a: Mapped[float] = mapped_column(Float, default=1.0)      # discrimination
    kind: Mapped[str] = mapped_column(String(30), default="practice")
    provenance: Mapped[str] = mapped_column(String(200), default="human")
    item_type: Mapped[str] = mapped_column(String(30), default="multiple_choice")
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    tolerance: Mapped[float | None] = mapped_column(Float, nullable=True)
    rubric: Mapped[str | None] = mapped_column(Text, nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)


class ItemConcept(Base):
    __tablename__ = "item_concepts"
    id: Mapped[int] = mapped_column(primary_key=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"))
    concept_id: Mapped[int] = mapped_column(ForeignKey("concepts.id"))
    is_primary: Mapped[bool] = mapped_column(Boolean, default=True)


class SessionRow(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    started_ts: Mapped[float] = mapped_column(Float)
    ended_ts: Mapped[float | None] = mapped_column(Float, nullable=True)


class StudentConceptState(Base):
    """Individual learner state (the online side of global/individual separation)."""
    __tablename__ = "student_concept_states"
    __table_args__ = (UniqueConstraint("user_id", "graph_version_id", "concept_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    graph_version_id: Mapped[int] = mapped_column(ForeignKey("concept_graph_versions.id"))
    concept_id: Mapped[int] = mapped_column(ForeignKey("concepts.id"))
    theta: Mapped[float] = mapped_column(Float)
    var: Mapped[float] = mapped_column(Float)
    succ: Mapped[int] = mapped_column(Integer, default=0)
    fail: Mapped[int] = mapped_column(Integer, default=0)
    stability: Mapped[float] = mapped_column(Float, default=1.0)
    last_success_ts: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_update_ts: Mapped[float | None] = mapped_column(Float, nullable=True)
    mastered: Mapped[bool] = mapped_column(Boolean, default=False)
    exposure: Mapped[int] = mapped_column(Integer, default=0)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    kind: Mapped[str] = mapped_column(String(40))             # global_model | baseline_bkt | ...
    params_json: Mapped[str] = mapped_column(Text)            # frozen global params at registration
    created_ts: Mapped[float] = mapped_column(Float)


class PolicyVersion(Base):
    __tablename__ = "policy_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    table_json: Mapped[str] = mapped_column(Text)             # rule-cell parameters at registration
    created_ts: Mapped[float] = mapped_column(Float)


class Intervention(Base):
    """One policy decision served to the student. exploration flag makes this
    table a valid behavior-policy log for off-policy evaluation later."""
    __tablename__ = "interventions"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    concept_id: Mapped[int | None] = mapped_column(ForeignKey("concepts.id"), nullable=True)
    item_id: Mapped[int | None] = mapped_column(ForeignKey("items.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(40))
    exploration: Mapped[bool] = mapped_column(Boolean, default=False)
    rationale: Mapped[str] = mapped_column(String(200))
    policy_version_id: Mapped[int] = mapped_column(ForeignKey("policy_versions.id"))
    ts: Mapped[float] = mapped_column(Float)


class Attempt(Base):
    """Telemetry contract (Phase 1 §1.3): every field is part of the data contract."""
    __tablename__ = "attempts"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"))
    intervention_id: Mapped[int | None] = mapped_column(ForeignKey("interventions.id"), nullable=True)
    choice_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    correct: Mapped[bool] = mapped_column(Boolean)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    attempt_no: Mapped[int] = mapped_column(Integer, default=1)
    hint_level_reached: Mapped[int] = mapped_column(Integer, default=0)
    paste_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_tag: Mapped[str | None] = mapped_column(String(40), nullable=True)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    grading_method: Mapped[str | None] = mapped_column(String(40), nullable=True)
    ts: Mapped[float] = mapped_column(Float)


class ModelPrediction(Base):
    """Append-only prediction log: every served decision pins model version, inputs,
    probabilities, and the eventual outcome (via intervention -> attempt)."""
    __tablename__ = "model_predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    intervention_id: Mapped[int] = mapped_column(ForeignKey("interventions.id"))
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    features_json: Mapped[str] = mapped_column(Text)
    p_correct: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision_json: Mapped[str] = mapped_column(Text)
    ts: Mapped[float] = mapped_column(Float)


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    design_json: Mapped[str] = mapped_column(Text, default="{}")
    started_ts: Mapped[float | None] = mapped_column(Float, nullable=True)
    ended_ts: Mapped[float | None] = mapped_column(Float, nullable=True)


class Assignment(Base):
    __tablename__ = "assignments"
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    arm: Mapped[str] = mapped_column(String(40))
    ts: Mapped[float] = mapped_column(Float)
