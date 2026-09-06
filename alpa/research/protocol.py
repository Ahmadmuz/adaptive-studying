"""Common evaluation protocol for knowledge-tracing models.

Interface contract — every model exposes:

    name: str
    fit(ds: KTDataset) -> None                      # may be a no-op
    predict_prefixes(skills, corrects) -> np.ndarray

`predict_prefixes` returns p_t for t = 1..T-1, where p_t predicts
`corrects[t]` using ONLY history `0..t-1`. The harness never hands a model
more than the prefix; `leakage.py` verifies this invariant empirically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, log_loss, roc_auc_score)

from .data import KTDataset

ECE_BINS = 15
COLD_START_K = 10          # cold-start = first K interactions of a student
MIN_MACRO_LEN = 20         # users eligible for per-student macro metrics


def _safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, n = 0.0, len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(ece)


@dataclass
class PredictionSet:
    """Per-user predictions for one model on one split. The unit of
    resampling for confidence intervals is the STUDENT, not the row."""
    y: list[np.ndarray] = field(default_factory=list)
    p: list[np.ndarray] = field(default_factory=list)
    users: list = field(default_factory=list)

    def add(self, user, yy, pp):
        self.users.append(user)
        self.y.append(np.asarray(yy, dtype=np.int64))
        self.p.append(np.asarray(pp, dtype=np.float64))

    @property
    def Y(self):
        return np.concatenate(self.y) if self.y else np.array([])

    @property
    def P(self):
        return np.concatenate(self.p) if self.p else np.array([])


def pooled_metrics(ps: PredictionSet, threshold: float = 0.5) -> dict:
    y, p = ps.Y, ps.P
    if len(y) == 0:
        return {}
    return {
        "n_predictions": int(len(y)),
        "roc_auc": float(_safe_auc(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, (p >= threshold).astype(int))),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "base_rate": float(y.mean()),
    }


def macro_student_auc(ps: PredictionSet, min_len: int = MIN_MACRO_LEN) -> dict:
    aucs = []
    for yy, pp in zip(ps.y, ps.p):
        if len(yy) >= min_len:
            a = _safe_auc(yy, pp)
            if not math.isnan(a):
                aucs.append(a)
    return {"macro_student_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "n_students_macro": len(aucs)}


def cold_start_metrics(ps: PredictionSet, k: int = COLD_START_K) -> dict:
    cs = PredictionSet()
    for u, yy, pp in zip(ps.users, ps.y, ps.p):
        if len(yy) > 1:
            cs.add(u, yy[: k - 1], pp[: k - 1])
    m = pooled_metrics(cs)
    m["cold_start_k"] = k
    return m


def evaluate_on_split(model, ds: KTDataset, indices) -> PredictionSet:
    ps = PredictionSet()
    for i in indices:
        skills, corrects = ds.seq(i)
        if len(skills) < 2:
            continue
        p = model.predict_prefixes(skills, corrects)
        assert len(p) == len(skills) - 1, \
            f"{model.name}: expected {len(skills)-1} predictions, got {len(p)}"
        assert np.isfinite(p).all(), f"{model.name}: non-finite predictions"
        ps.add(ds.users[i], corrects[1:], np.clip(p, 1e-7, 1 - 1e-7))
    return ps


def _resample_users(ps: PredictionSet, rng) -> PredictionSet:
    idx = rng.integers(0, len(ps.users), size=len(ps.users))
    out = PredictionSet()
    for i in idx:
        out.add(ps.users[i], ps.y[i], ps.p[i])
    return out


def bootstrap_ci(ps: PredictionSet, n_boot: int = 200, seed: int = 0,
                 metrics=("roc_auc", "log_loss", "accuracy")) -> dict:
    rng = np.random.default_rng(seed)
    dist = {m: [] for m in metrics}
    for _ in range(n_boot):
        m = pooled_metrics(_resample_users(ps, rng))
        for k in metrics:
            if k in m and math.isfinite(m[k]):
                dist[k].append(m[k])
    return {f"{k}_ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in dist.items() if v}


def paired_auc_delta(ps_a: PredictionSet, ps_b: PredictionSet,
                     n_boot: int = 200, seed: int = 0) -> dict:
    """Bootstrap distribution of AUC(A) - AUC(B) on matched students.
    Students present in both sets are aligned by user id."""
    map_b = {u: (y, p) for u, y, p in zip(ps_b.users, ps_b.y, ps_b.p)}
    pairs = [(u, y, p, map_b[u][0], map_b[u][1])
             for u, y, p in zip(ps_a.users, ps_a.y, ps_a.p) if u in map_b]
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(pairs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ya = np.concatenate([pairs[i][1] for i in idx])
        pa = np.concatenate([pairs[i][2] for i in idx])
        yb = np.concatenate([pairs[i][3] for i in idx])
        pb = np.concatenate([pairs[i][4] for i in idx])
        aa, ab = _safe_auc(ya, pa), _safe_auc(yb, pb)
        if not (math.isnan(aa) or math.isnan(ab)):
            deltas.append(aa - ab)
    deltas = np.asarray(deltas)
    return {"mean_delta_auc": float(deltas.mean()),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "p_worse": float((deltas < 0).mean()), "n_students": n}


def full_report(model, ds: KTDataset, split: str = "test", n_boot: int = 200,
                seed: int = 0) -> dict:
    ps = evaluate_on_split(model, ds, ds.subset(split))
    rep = {"model": model.name, "split": split}
    rep.update(pooled_metrics(ps))
    rep.update(macro_student_auc(ps))
    rep["cold_start"] = cold_start_metrics(ps)
    rep.update(bootstrap_ci(ps, n_boot=n_boot, seed=seed))
    return rep, ps
