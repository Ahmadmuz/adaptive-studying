"""Dataset loading, preprocessing, student-level splits, and manifest.

Produces a CSR-style store (data arrays + per-user offsets) that every model
consumes, so all models see exactly the same preprocessed interactions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW = DATA_DIR / "raw" / "assistments2009.parquet"
PROCESSED = DATA_DIR / "processed" / "assistments2009.npz"
MANIFEST = DATA_DIR / "processed" / "assistments2009.manifest.json"

SPLITS = ("train", "val", "test")


@dataclass
class KTDataset:
    users: np.ndarray          # user ids, aligned with offsets
    offsets: np.ndarray        # len(users)+1 offsets into skills/corrects
    skills: np.ndarray         # int skill ids
    corrects: np.ndarray       # 0/1 outcomes
    skill_names: list[str]
    n_skills: int
    splits: dict[str, np.ndarray]   # split name -> indices into users
    meta: dict

    def __len__(self):
        return len(self.users)

    def seq(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return self.skills[lo:hi], self.corrects[lo:hi]

    def subset(self, split: str) -> list[int]:
        return list(self.splits[split])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(min_user_len: int = 5, min_skill_count: int = 100,
                 split_ratios=(0.70, 0.15, 0.15), seed: int = 42) -> KTDataset:
    if PROCESSED.exists() and MANIFEST.exists():
        return _load_cached()
    df = pd.read_parquet(RAW)
    rows = []
    for uid, skills, grades in zip(df["user_id"], df["skill_ids"], df["grades"]):
        for order, (sk, g) in enumerate(zip(skills, grades)):
            rows.append((uid, order, str(sk), int(g)))
    long = pd.DataFrame(rows, columns=["user_id", "order", "skill", "correct"])

    long = long.groupby("user_id").filter(lambda g: len(g) >= min_user_len)
    counts = long["skill"].value_counts()
    keep = set(counts[counts >= min_skill_count].index)
    long = long[long["skill"].isin(keep)]
    # dropping skills can shorten users; re-filter once
    long = long.groupby("user_id").filter(lambda g: len(g) >= min_user_len)

    skill_names = sorted(long["skill"].unique())
    skill_to_id = {s: i for i, s in enumerate(skill_names)}
    long["skill_id"] = long["skill"].map(skill_to_id)
    long = long.sort_values(["user_id", "order"], kind="mergesort")

    users = long["user_id"].unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(users))
    n = len(users)
    n_tr = int(n * split_ratios[0])
    n_va = int(n * split_ratios[1])
    split_of_user = {}
    for idx, u in zip(perm[:n_tr], users[perm[:n_tr]]):
        split_of_user[u] = "train"
    for u in users[perm[n_tr:n_tr + n_va]]:
        split_of_user[u] = "val"
    for u in users[perm[n_tr + n_va:]]:
        split_of_user[u] = "test"

    offsets, u_list, splits_idx = [0], [], {s: [] for s in SPLITS}
    skills_all, corrects_all = [], []
    for u in users:
        g = long[long["user_id"] == u]
        skills_all.append(g["skill_id"].to_numpy(dtype=np.int32))
        corrects_all.append(g["correct"].to_numpy(dtype=np.int8))
        u_list.append(u)
        splits_idx[split_of_user[u]].append(len(u_list) - 1)
        offsets.append(offsets[-1] + len(g))

    ds = KTDataset(
        users=np.asarray(u_list), offsets=np.asarray(offsets, dtype=np.int64),
        skills=np.concatenate(skills_all), corrects=np.concatenate(corrects_all),
        skill_names=skill_names, n_skills=len(skill_names),
        splits={s: np.asarray(v, dtype=np.int64) for s, v in splits_idx.items()},
        meta={
            "source": "huggingface:Atomi/ASSISTments2009 (ASSISTments 2009 skill-builder variant)",
            "raw_sha256": _sha256(RAW),
            "n_users": int(len(users)), "n_interactions": int(len(long)),
            "n_skills": len(skill_names),
            "wall_clock_timestamps": False,
            "split_policy": "student-level, seeded shuffle",
            "split_sizes": {s: len(v) for s, v in splits_idx.items()},
            "min_user_len": min_user_len, "min_skill_count": min_skill_count,
            "seed": seed,
        },
    )
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROCESSED, users=ds.users, offsets=ds.offsets, skills=ds.skills,
        corrects=ds.corrects,
        **{f"split_{s}": ds.splits[s] for s in SPLITS})
    MANIFEST.write_text(json.dumps({**ds.meta, "skill_names": skill_names}, indent=2))
    return ds


def _load_cached() -> KTDataset:
    z = np.load(PROCESSED)
    meta = json.loads(MANIFEST.read_text())
    skill_names = meta.pop("skill_names")
    return KTDataset(
        users=z["users"], offsets=z["offsets"], skills=z["skills"],
        corrects=z["corrects"], skill_names=skill_names,
        n_skills=len(skill_names),
        splits={s: z[f"split_{s}"] for s in SPLITS}, meta=meta)
