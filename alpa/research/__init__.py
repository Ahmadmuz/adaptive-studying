"""Knowledge-tracing research package.

Evaluation protocol rules (Phase A contract):
- student-level splits only (no interaction-level splits: they leak student identity);
- a prediction at position t may use only interactions 0..t-1;
- all fitting happens on train students; validation students gate early stopping;
  test students are touched exactly once for final numbers;
- this dataset variant carries NO wall-clock timestamps; any retention analysis
  on it uses step-lag as a proxy and must be labeled as such.
"""
