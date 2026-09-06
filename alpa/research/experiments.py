"""Experiment registry + runner.

Every run appends one JSON line to experiments/registry.jsonl — runs are never
overwritten. Reports (markdown) are written alongside with a timestamped name.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import leakage, protocol
from .data import KTDataset

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments"
REGISTRY = EXP_DIR / "registry.jsonl"
REPORTS = EXP_DIR / "reports"
CHECKPOINTS = EXP_DIR / "checkpoints"


def registry_record(model_name: str, ds: KTDataset, config: dict, metrics: dict,
                    notes: str = "", limitations: str = "") -> dict:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "experiment_id": f"kt-{model_name}-{int(time.time())}",
        "model": model_name,
        "dataset": ds.meta.get("source", "unknown"),
        "dataset_sha256": ds.meta.get("raw_sha256", ""),
        "split_policy": ds.meta.get("split_policy"),
        "split_sizes": ds.meta.get("split_sizes"),
        "features": config.get("features", "see model class"),
        "seed": config.get("seed"),
        "hyperparameters": config,
        "metrics": metrics,
        "checkpoint": config.get("checkpoint_path"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "notes": notes,
        "limitations": limitations,
        "status": "VALIDATED" if metrics else "FAILED",
    }
    with open(REGISTRY, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def run_leakage_suite(model, ds: KTDataset) -> list[dict]:
    return [
        leakage.future_perturbation_test(model, ds),
        leakage.outcome_independence_test(model, ds),
    ]


def evaluate_model(model, ds: KTDataset, config: dict, notes: str = "",
                   limitations: str = "", n_boot: int = 200,
                   run_leakage: bool = True) -> dict:
    t0 = time.time()
    model.fit(ds)
    leak = run_leakage_suite(model, ds) if run_leakage else []
    report, ps_test = protocol.full_report(model, ds, "test", n_boot=n_boot)
    val = protocol.pooled_metrics(protocol.evaluate_on_split(model, ds, ds.subset("val")))
    report["val_logloss"] = val.get("log_loss")
    report["leakage"] = leak
    report["fit_and_eval_seconds"] = round(time.time() - t0, 1)
    registry_record(model.name, ds, config, report, notes=notes, limitations=limitations)
    return report


def write_report(title: str, rows: list[dict], extra_md: str = "") -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"kt-benchmark-{time.strftime('%Y%m%d-%H%M%S')}.md"
    cols = ["model", "roc_auc", "pr_auc", "accuracy", "log_loss", "brier", "ece",
            "macro_student_auc", "fit_and_eval_seconds"]
    lines = [f"# {title}", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Student-level held-out "
             "test split; all models fit on train students only.", "",
             "| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for r in rows:
        def fmt(k):
            v = r.get(k)
            return "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(fmt(c) for c in cols) + " |")
    if extra_md:
        lines += ["", extra_md]
    path.write_text("\n".join(lines) + "\n")
    return path
