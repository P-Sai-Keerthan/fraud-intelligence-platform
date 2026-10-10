# Sequence learning for early fraud detection (Experiment 2)

**Status: Phase 1 — audit and protocol draft. No model trained, no results.**

| File | What it is |
|---|---|
| [`AUDIT.md`](AUDIT.md) | What the existing generator, features, splits and metrics do; what limits the earlier comparison; power analysis; environment limits; public-dataset suitability. |
| [`PROTOCOL_DRAFT.md`](PROTOCOL_DRAFT.md) | Proposed models, inputs, endpoints, splits, tuning budgets, seeds, metrics, leakage controls. **Not frozen.** Becomes `PROTOCOL.md` plus `FREEZE.json` and a git tag only after the project owner approves. |

Nothing here modifies the production application or any completed experiment. The new experiment depends on
the v2 generator and evaluation code from `origin/Keerthan` (see AUDIT §0 for the branch question) and on the
files of `experiments/lstm_value_ablation/`, which were not available when this was written.
