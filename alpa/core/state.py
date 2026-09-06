"""Persistent, versioned student state (Milestone 3).

Design contract:
- mastery and uncertainty are SEPARATE quantities (mastered flag / p_mastery vs
  posterior variance); a student can be mastered=True with high uncertainty.
- serializable to/from JSON with an explicit schema version;
- the state transition is deterministic given (state, observation, model params):
  no clocks are read internally — the caller supplies `now`;
- the DB remains the system of record in the API path; this module is the
  canonical in-memory representation and the unit of reproducibility tests.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from . import model
from .types import ConceptSnapshot, GlobalParams, ItemSpec

STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Observation:
    concept_id: int
    correct: bool
    ts: float
    item_a: float = 1.0
    item_b: float = 0.0


@dataclass
class StudentState:
    user_id: int
    graph_version: int
    schema_version: int = STATE_SCHEMA_VERSION
    concepts: dict[int, ConceptSnapshot] = field(default_factory=dict)

    # ------------------------------------------------------------ transitions
    def apply(self, obs: Observation, gp: GlobalParams) -> "StudentState":
        """Deterministic transition S_{t+1} = F(S_t, o). Returns self (mutated)."""
        snap = self.concepts[obs.concept_id]
        item = ItemSpec(item_id=obs.concept_id, stem="", choices=("a", "b"),
                        correct_index=0, concept_ids=(obs.concept_id,),
                        b=obs.item_b, a=obs.item_a)
        new = model.update(snap, item, 1 if obs.correct else 0, obs.ts, gp)
        new.mastered = model.apply_mastery_hysteresis(new, obs.ts, gp)
        self.concepts[obs.concept_id] = new
        return self

    def mastery(self, concept_id: int, now: float, gp: GlobalParams) -> float:
        return model.mastery_prob(self.concepts[concept_id], now, gp)

    def uncertainty(self, concept_id: int, now: float, gp: GlobalParams) -> float:
        return float(model.eff_var(self.concepts[concept_id], now, gp)) ** 0.5

    def retention(self, concept_id: int, now: float, gp: GlobalParams) -> float:
        return model.retention(self.concepts[concept_id], now, gp)

    # ---------------------------------------------------------- serialization
    def to_json(self) -> str:
        return json.dumps({
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "graph_version": self.graph_version,
            "concepts": {str(k): asdict(v) for k, v in sorted(self.concepts.items())},
        }, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "StudentState":
        d = json.loads(raw)
        if d.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported state schema {d.get('schema_version')}")
        concepts = {int(k): ConceptSnapshot(**v) for k, v in d["concepts"].items()}
        return cls(user_id=d["user_id"], graph_version=d["graph_version"],
                   schema_version=d["schema_version"], concepts=concepts)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    # ------------------------------------------------------------- factories
    @classmethod
    def from_priors(cls, user_id: int, graph_version: int,
                    concept_defs: list[tuple[int, str, float, float]]) -> "StudentState":
        """concept_defs: (concept_id, code, mu, sigma) — cold-start initialization."""
        return cls(user_id=user_id, graph_version=graph_version, concepts={
            cid: ConceptSnapshot(concept_id=cid, code=code, theta=mu, var=sigma ** 2)
            for cid, code, mu, sigma in concept_defs})


def states_equal(a: StudentState, b: StudentState, tol: float = 0.0) -> bool:
    """Reproducibility comparison. With tol=0 this is an exact deterministic check."""
    if (a.user_id, a.graph_version, a.schema_version) != \
       (b.user_id, b.graph_version, b.schema_version):
        return False
    if set(a.concepts) != set(b.concepts):
        return False
    for cid, sa in a.concepts.items():
        sb = b.concepts[cid]
        for f in ("theta", "var", "stability"):
            if abs(getattr(sa, f) - getattr(sb, f)) > tol:
                return False
        if (sa.succ, sa.fail, sa.mastered, sa.exposure,
            sa.last_success_ts, sa.last_update_ts) != \
           (sb.succ, sb.fail, sb.mastered, sb.exposure,
            sb.last_success_ts, sb.last_update_ts):
            return False
    return True
