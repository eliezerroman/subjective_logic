# experiments/

Research-methodology code built on top of the `subjective_logic` library,
used across the experiment notebooks in `notebooks/` (Phases 0-6 of the
validation plan). This is deliberately kept separate from
`src/subjective_logic/`: the library implements Jøsang (2016)'s formulas
and is validated against the book; this folder implements *our own*
methodological choices (how to turn a classifier's softmax, an LLM's
samples, or an RL agent's visit counts into `(r, s, a)`, and how to check
whether the resulting opinions are honest) and is validated against
the experiments themselves.

## Contents

- `common.py` — helper functions shared across phases: `credible_interval`
  (Beta credible interval for an opinion, from its equivalent evidence)
  and `standardized_sq_error` (per-observation calibration check for an
  opinion's stated uncertainty, used to compute MSSE across many runs).
  Introduced and validated in Phase 0 (`notebooks/01_phase0_synthetic.ipynb`).

More modules will be added here as later phases introduce their own
reusable pieces (e.g. evidence-extraction functions for classifiers,
LLMs, and RL agents).