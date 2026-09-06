"""Synthetic student simulator with PLANTED dynamics (Phase L foundation).

Generates interaction sequences over the mechanics prerequisite graph where the
true learning dynamics are known by construction: ability grows with practice
(faster when prerequisites are solid), decays with step-lag (forgetting), and
profiles differ in rate, prior, noise, and concept-specific weakness.

Honesty contract: results on this dataset validate that the MODELS AND POLICY
recover planted structure. They are NEVER evidence about human learning.
Timestamps are synthetic session steps (no wall clock).
"""
from __future__ import annotations

import numpy as np

from ..seed.mechanics import CONCEPTS
from .data import KTDataset


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


PROFILES = {
    "fast":            dict(learn=0.50, prior=0.0, forget=0.00, noise=0.5),
    "slow":            dict(learn=0.12, prior=0.0, forget=0.00, noise=0.5),
    "high_prior":      dict(learn=0.30, prior=1.2, forget=0.00, noise=0.5),
    "low_prior":       dict(learn=0.30, prior=-1.2, forget=0.00, noise=0.5),
    "forgetting":      dict(learn=0.35, prior=0.0, forget=0.30, noise=0.5),
    "inconsistent":    dict(learn=0.30, prior=0.0, forget=0.00, noise=1.6),
}


def generate_synthetic(n_students: int = 1200, n_interactions: int = 60,
                       seed: int = 7) -> KTDataset:
    rng = np.random.default_rng(seed)
    codes = [c[0] for c in CONCEPTS]
    prereq_idx = {i: [codes.index(p) for p in c[3]] for i, c in enumerate(CONCEPTS)}
    S = len(codes)
    b = np.linspace(-0.8, 1.2, S)           # planted difficulty, ordered by topo

    users, offsets, skills_all, corr_all = [], [0], [], []
    split_idx = {"train": [], "val": [], "test": []}
    perm = rng.permutation(n_students)
    n_tr = int(0.7 * n_students)
    n_va = int(0.15 * n_students)
    split_of = {}
    for k, u in enumerate(perm):
        split_of[u] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")

    for u in range(n_students):
        pname = list(PROFILES)[u % len(PROFILES)]
        pr = PROFILES[pname]
        # concept-specific weakness for ~15% of students
        weak = set(rng.choice(S, size=3, replace=False)) if u % 7 == 0 else set()
        theta = np.full(S, pr["prior"] + rng.normal(0, 0.3))
        if weak:
            theta[list(weak)] -= 1.0
        last = np.full(S, -10)
        sk_seq, y_seq = [], []
        frontier = [0]
        for t in range(n_interactions):
            # policy-like curriculum: prefer frontier concepts, sometimes review
            avail = [c for c in range(S)
                     if all(theta[p] > b[p] - 0.3 for p in prereq_idx[c])]
            if not avail:
                avail = list(range(S))
            if rng.random() < 0.25 and sk_seq:
                s = int(rng.choice(sk_seq[-15:]))            # review recent
            else:
                s = int(rng.choice(avail))
            gap = t - last[s]
            theta_eff = theta[s] - pr["forget"] * np.log1p(max(gap, 0))
            prereq_bonus = 0.4 * np.mean([theta[p] > b[p] for p in prereq_idx[s]]) \
                if prereq_idx[s] else 0.0
            z = theta_eff - b[s] + rng.normal(0, pr["noise"]) + prereq_bonus
            y = int(rng.random() < _sigmoid(z))
            sk_seq.append(s)
            y_seq.append(y)
            last[s] = t
            if y == 1:
                theta[s] += pr["learn"] * (1.0 + prereq_bonus)
            else:
                theta[s] -= 0.3 * pr["learn"]
        users.append(u)
        skills_all.append(np.asarray(sk_seq, dtype=np.int32))
        corr_all.append(np.asarray(y_seq, dtype=np.int8))
        split_idx[split_of[u]].append(len(users) - 1)
        offsets.append(offsets[-1] + len(sk_seq))

    return KTDataset(
        users=np.asarray(users), offsets=np.asarray(offsets, dtype=np.int64),
        skills=np.concatenate(skills_all), corrects=np.concatenate(corr_all),
        skill_names=codes, n_skills=S,
        splits={k: np.asarray(v, dtype=np.int64) for k, v in split_idx.items()},
        meta={"source": "synthetic:alpa.research.synthetic (planted dynamics)",
              "wall_clock_timestamps": False, "synthetic": True,
              "split_policy": "student-level, seeded", "seed": seed,
              "n_users": n_students, "n_interactions": int(offsets[-1]),
              "n_skills": S,
              "profiles": list(PROFILES),
              "split_sizes": {k: len(v) for k, v in split_idx.items()}})
