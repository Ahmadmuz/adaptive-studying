#!/usr/bin/env python3
"""Knowledge-tracing benchmark runner (Phase A).

    python scripts/run_kt_eval.py --stage shallow     # fast numpy/sklearn models
    python scripts/run_kt_eval.py --stage torch       # GRU-KT + attention-KT
    python scripts/run_kt_eval.py --stage synthetic   # planted-dynamics checks

All runs are registry-recorded; test-split metrics use student-level holdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpa.research import experiments as exp
from alpa.research.data import load_dataset
from alpa.research.models import (BKTModel, DAKT, GRUKT, HybridProduction, IRTModel,
                                  LogisticRich, MajorityBaseline, PFALogistic,
                                  RandomBaseline, SkillMajorityBaseline, device)


def shallow():
    ds = load_dataset()
    print("dataset:", ds.meta["n_users"], "users,", ds.meta["n_interactions"],
          "interactions,", ds.meta["n_skills"], "skills")
    models = [
        (RandomBaseline(seed=0), {"seed": 0}, "uniform random probabilities", ""),
        (MajorityBaseline(), {}, "global train base rate", ""),
        (SkillMajorityBaseline(), {}, "per-skill train frequency", ""),
        (PFALogistic(C=1.0), {"C": 1.0}, "PFA skill-level counts, L2 logistic",
         "skill KCs, no item ids; counts are unbounded features"),
        (LogisticRich(C=1.0), {"C": 1.0},
         "PFA counts + recency/accuracy/position features", "same"),
        (BKTModel(n_iter=25), {"n_iter": 25}, "per-skill BKT, EM fit on train",
         "independence of skills given per-skill latent state"),
        (IRTModel(n_steps=300, lr=0.05), {"n_steps": 300, "lr": 0.05, "l2": 1e-3},
         "2PL IRT, skills as items, static ability",
         "no learning dynamics; test students get theta=0 prior (cold-start handicap)"),
        (HybridProduction(), {"params": "GlobalParams defaults",
                              "item_b": "calibrated from train skill rates", "item_a": 1.0},
         "production PFA/IRT hybrid with forgetting term (inactive: no timestamps)",
         "item bank difficulty collapsed to per-skill b"),
    ]
    rows = []
    for model, cfg, notes, lim in models:
        rep = exp.evaluate_model(model, ds, cfg, notes=notes, limitations=lim, n_boot=200)
        print(f"{model.name:22s} AUC={rep['roc_auc']:.4f} logloss={rep['log_loss']:.4f} "
              f"ECE={rep['ece']:.4f} macro={rep['macro_student_auc']:.4f} "
              f"leak_ok={all(l['passed'] for l in rep['leakage'])} "
              f"({rep['fit_and_eval_seconds']}s)")
        rows.append(rep)
    return rows, ds


def torch_stage():
    ds = load_dataset()
    print("torch device:", device())
    rows = []
    for cls, cfg in [
        (GRUKT, dict(d_model=64, epochs=12, lr=1e-3, batch=32, chunk=200, seed=42)),
        (DAKT, dict(d_model=64, epochs=10, lr=1e-3, batch=32, chunk=200, seed=42)),
    ]:
        m = cls(checkpoint_dir=str(exp.CHECKPOINTS), **cfg)
        rep = exp.evaluate_model(
            m, ds, {**cfg, "checkpoint_path": getattr(m, "checkpoint_path", None)},
            notes=f"{m.name} trained on train students, early-stopped on val logloss",
            limitations="CPU training; d_model 64; no item-difficulty features",
            n_boot=200, run_leakage=True)
        print(f"{m.name:22s} AUC={rep['roc_auc']:.4f} logloss={rep['log_loss']:.4f} "
              f"ECE={rep['ece']:.4f} macro={rep['macro_student_auc']:.4f} "
              f"val_ll={rep['val_logloss']:.4f} ({rep['fit_and_eval_seconds']}s)")
        rows.append(rep)
    return rows, ds


def synthetic_stage():
    from alpa.research.models import ConceptAwareGRU
    from alpa.research.synthetic import generate_synthetic
    from alpa.seed.mechanics import CONCEPTS
    ds = generate_synthetic()
    print("synthetic:", ds.meta["n_users"], "users,", ds.meta["n_interactions"], "interactions")
    codes = [c[0] for c in CONCEPTS]
    prereqs = {i: [codes.index(p) for p in c[3]] for i, c in enumerate(CONCEPTS)}
    rows = []
    entries = [
        (MajorityBaseline(), {}, "base-rate reference on planted data", False),
        (PFALogistic(C=1.0), {"C": 1.0}, "PFA on planted data", False),
        (BKTModel(n_iter=25), {"n_iter": 25}, "BKT on planted data", False),
        (HybridProduction(), {}, "production hybrid on planted data", False),
        (GRUKT(d_model=48, epochs=10, lr=1e-3, batch=32, chunk=200, seed=42,
               checkpoint_dir=str(exp.CHECKPOINTS)),
         {"d_model": 48, "epochs": 10}, "GRU-KT on planted data (no graph signal)", True),
        (ConceptAwareGRU(prereqs, len(codes), d_model=48, epochs=10, lr=1e-3,
                         batch=32, chunk=200, seed=42,
                         checkpoint_dir=str(exp.CHECKPOINTS)),
         {"d_model": 48, "epochs": 10, "prereqs": "mechanics graph"},
         "GRU-KT + given prerequisite running rates", True),
    ]
    for model, cfg, notes, is_torch in entries:
        rep = exp.evaluate_model(model, ds, cfg, notes=notes,
                                 limitations="synthetic; not human evidence", n_boot=100)
        print(f"{model.name:22s} AUC={rep['roc_auc']:.4f} logloss={rep['log_loss']:.4f} "
              f"ECE={rep['ece']:.4f} ({rep['fit_and_eval_seconds']}s)")
        rows.append(rep)
    return rows, ds


def analysis_stage():
    """Paired comparisons + cold-start table on the real dataset."""
    from alpa.research import protocol
    ds = load_dataset()
    models = {
        "pfa_logistic": PFALogistic(C=1.0),
        "logistic_rich": LogisticRich(C=1.0),
        "bkt_em": BKTModel(n_iter=25),
        "hybrid_production": HybridProduction(),
        "skill_majority": SkillMajorityBaseline(),
    }
    for m in models.values():
        m.fit(ds)
    ps = {name: protocol.evaluate_on_split(m, ds, ds.subset("test"))
          for name, m in models.items()}
    print("== paired AUC deltas (bootstrap over students, test split) ==")
    for a, b in [("logistic_rich", "hybrid_production"),
                 ("logistic_rich", "pfa_logistic"),
                 ("pfa_logistic", "bkt_em"),
                 ("hybrid_production", "skill_majority")]:
        d = protocol.paired_auc_delta(ps[a], ps[b], n_boot=200, seed=0)
        print(f"AUC({a}) - AUC({b}) = {d['mean_delta_auc']:+.4f} "
              f"CI95 [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
              f"P(worse)={d['p_worse']:.3f} n={d['n_students']}")
    print("== cold-start (first 9 predictions per student) ==")
    for name, p in ps.items():
        cs = protocol.cold_start_metrics(p)
        print(f"{name:22s} AUC={cs['roc_auc']:.4f} logloss={cs['log_loss']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",
                    choices=["shallow", "torch", "synthetic", "analysis", "all"],
                    default="shallow")
    args = ap.parse_args()
    all_rows = []
    if args.stage in ("shallow", "all"):
        rows, ds = shallow()
        all_rows += rows
    if args.stage in ("torch", "all"):
        rows, ds = torch_stage()
        all_rows += rows
    if args.stage in ("synthetic", "all"):
        rows, ds = synthetic_stage()
        all_rows += rows
    if args.stage == "analysis":
        analysis_stage()
    if all_rows:
        path = exp.write_report(f"KT benchmark — stage {args.stage}", all_rows)
        print("report:", path)


if __name__ == "__main__":
    main()
