"""Leakage probes and future-perturbation tests.

These tests make the no-future-information invariant EMPIRICAL rather than
asserted: they perturb or truncate the future of a sequence and require every
prediction up to position t to be bit-identical.
"""
from __future__ import annotations

import numpy as np

from .data import KTDataset


def future_perturbation_test(model, ds: KTDataset, n_users: int = 30,
                             tol: float = 1e-9, seed: int = 0) -> dict:
    """For sampled users and sampled cut points t, predictions on the prefix
    [:t] must equal predictions on the full sequence at positions < t, AND
    must be invariant to arbitrary corruption of positions >= t."""
    rng = np.random.default_rng(seed)
    checked = violations = 0
    idxs = rng.choice(len(ds.users), size=min(n_users, len(ds.users)), replace=False)
    for i in idxs:
        skills, corrects = ds.seq(i)
        T = len(skills)
        if T < 6:
            continue
        t = int(rng.integers(3, T - 1))
        # predictions at positions 1..t-1 use history up to index t-2, so they
        # must be identical across the full, truncated, and corrupted sequences
        p_full = model.predict_prefixes(skills, corrects)[: t - 1]
        p_prefix = model.predict_prefixes(skills[:t], corrects[:t])
        corrupted = corrects.copy()
        corrupted[t:] = 1 - corrupted[t:]                    # flip the future
        shuffled_skills = skills.copy()
        rng.shuffle(shuffled_skills[t:])                     # scramble the future
        p_corrupt = model.predict_prefixes(shuffled_skills, corrupted)[: t - 1]
        checked += 2
        if not (np.allclose(p_full, p_prefix, atol=tol)
                and np.allclose(p_full, p_corrupt, atol=tol)):
            violations += 1
    return {"test": "future_perturbation", "model": model.name,
            "checked": checked, "violations": violations,
            "passed": violations == 0}


def feature_prefix_audit(build_features, ds: KTDataset, n_users: int = 20,
                         tol: float = 1e-12, seed: int = 1) -> dict:
    """For feature-based models: features computed for position t must be
    identical whether the builder sees the full sequence or only the prefix."""
    rng = np.random.default_rng(seed)
    checked = violations = 0
    idxs = rng.choice(len(ds.users), size=min(n_users, len(ds.users)), replace=False)
    for i in idxs:
        skills, corrects = ds.seq(i)
        T = len(skills)
        if T < 6:
            continue
        t = int(rng.integers(2, T - 1))
        X_full = build_features(skills, corrects)
        X_prefix = build_features(skills[:t + 1], corrects[:t + 1])
        checked += 1
        if not np.allclose(X_full[:t], X_prefix[:t], atol=tol):
            violations += 1
    return {"test": "feature_prefix_audit", "checked": checked,
            "violations": violations, "passed": violations == 0}


def outcome_independence_test(model, ds: KTDataset, n_users: int = 30,
                              tol: float = 1e-9, seed: int = 2) -> dict:
    """Prediction at t must not depend on y_t itself (the current-row outcome).
    Flipping y_t while keeping history and future fixed must not move p_t."""
    rng = np.random.default_rng(seed)
    checked = violations = 0
    idxs = rng.choice(len(ds.users), size=min(n_users, len(ds.users)), replace=False)
    for i in idxs:
        skills, corrects = ds.seq(i)
        T = len(skills)
        if T < 5:
            continue
        t = int(rng.integers(1, T - 1))
        flipped = corrects.copy()
        flipped[t] = 1 - flipped[t]
        p_a = model.predict_prefixes(skills, corrects)
        p_b = model.predict_prefixes(skills, flipped)
        checked += 1
        # positions up to and including t use history strictly before themselves
        if not np.allclose(p_a[:t], p_b[:t], atol=tol):
            violations += 1
    return {"test": "outcome_independence", "model": model.name,
            "checked": checked, "violations": violations,
            "passed": violations == 0}
