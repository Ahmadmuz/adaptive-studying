# Phase A — Knowledge Tracing: Consolidated Benchmark Report

**Date:** 2026-09-06 · **Status:** VALIDATED (all numbers from executed runs)
**Registry:** `experiments/registry.jsonl` (append-only, one record per model-run)

## Dataset

- **Source:** ASSISTments 2009 skill-builder variant
  (`huggingface:Atomi/ASSISTments2009`), raw sha256 pinned in
  `data/processed/assistments2009.manifest.json`.
- **After preprocessing:** 3,620 students · 272,368 interactions · 114 skills
  (filters: users ≥5 interactions, skills ≥100 occurrences).
- **Split:** student-level 2534 / 543 / 543 (train/val/test), seeded. All fitting
  on train students; test touched once for final numbers.
- **Timestamps:** this variant has NO wall-clock timestamps. Any retention claim
  on it would use step-lag as a proxy only; no forgetting analysis is reported
  here for that reason.
- Base rate (train): 0.662.

## Protocol

- Prediction at position t uses history 0..t-1 only; verified empirically per
  model by `future_perturbation` and `outcome_independence` probes (all pass).
- Metrics: ROC-AUC, PR-AUC, accuracy, log loss, Brier, ECE (15 bins),
  macro per-student AUC (users ≥20 predictions), cold-start (first 9
  predictions), bootstrap CIs (user-cluster resampling, n=200), paired AUC deltas.

## Main results (test split)

| model | ROC-AUC | logloss | ECE | macro-AUC | cold-start AUC |
|---|---|---|---|---|---|
| random | 0.500 | 0.990 | 0.258 | 0.511 | - |
| majority_global | 0.500 | 0.659 | 0.033 | 0.500 | - |
| majority_skill | 0.632 | 0.632 | 0.031 | 0.610 | 0.640 |
| pfa_logistic | 0.701 | 0.601 | 0.027 | 0.606 | 0.720 |
| logistic_rich | **0.722** | **0.582** | **0.018** | 0.602 | **0.732** |
| bkt_em | 0.698 | 0.597 | 0.020 | 0.603 | 0.725 |
| irt_2pl | 0.542 | 1.578 | 0.232 | 0.530 | - |
| hybrid_production (untuned) | 0.675 | 0.630 | 0.073 | 0.565 | 0.681 |
| gru_kt | **0.748** | 0.562 | 0.030 | **0.656** | - |
| dakt_attention | 0.722 | 0.583 | 0.029 | 0.645 | - |

IRT caveat: static abilities, no learning dynamics; test students get the θ=0
prior (honest cold-start handicap). Its role is as an anchor, not a competitor.

## Paired comparisons (bootstrap over matched students, test split)

- logistic_rich − hybrid_production: **+0.047 AUC, CI [+0.039, +0.053], P(worse)=0.000**
- logistic_rich − pfa_logistic: +0.021, CI [+0.017, +0.025]
- pfa_logistic − bkt_em: +0.003, CI [−0.002, +0.007] → statistically indistinguishable
- hybrid_production − skill_majority: +0.044, CI [+0.033, +0.056]

## Synthetic (planted dynamics — NOT human evidence)

1,200 students / 72,000 interactions over the mechanics prerequisite graph:
majority 0.500 · PFA 0.728 · BKT 0.709 · hybrid 0.703 · GRU-KT 0.737 ·
concept-aware GRU (given prereq graph) 0.736.

**Negative result, reported as-is:** supplying the prerequisite graph as running
prereq-success-rate features did not improve the GRU on this planted data.
Likely causes: the planted prereq coupling (0.4 logit bonus) is small vs. the
noise floor, and the GRU already extracts ordering structure implicitly. The
graph channel must be re-tested with stronger planted coupling before claiming
any benefit either way.

## The 0.90 target: honest verdict

**Not reached, and not expected under this protocol.** The historically reported
DKT AUCs ≥0.90 on ASSISTments were produced under interaction-level splits and
target-inflating evaluation (documented in the KT reassessment literature). Under
a strict student-level split with prefix-only prediction, 0.72–0.75 AUC is the
state we can validate here; this is consistent with published reassessments.
We report 0.748 (GRU-KT) as the strongest validated number and will not tune on
the test set to close the gap.

## Findings with engineering consequences

1. **The production hybrid is measurably weaker than logistic-rich on raw
   next-response prediction** (+0.047 AUC, significant). Diagnosis: untuned
   defaults, fixed discrimination a=1, no recency features, forgetting term
   inactive (no timestamps). Action queued: add recency features and fit item
   discrimination offline; keep the interpretable θ/mastery path intact —
   prediction AUC is not the production objective, but a 5-point gap deserves a fix.
2. PFA ≈ BKT here (ΔAUC 0.003, CI straddles 0) — the Phase 1 decision to start
   interpretable is vindicated; complexity must earn its place.
3. Calibration is good for the logistic family (ECE ≤0.03); the hybrid's ECE
   0.073 needs the same fix as (1).
4. All models pass the leakage probes; the harness itself is trustworthy.

## Reproducibility

`python scripts/run_kt_eval.py --stage {shallow|torch|synthetic|analysis}` —
seeds pinned (42 for torch models), dataset hash pinned, checkpoints under
`experiments/checkpoints/`, every run appended to the registry.
