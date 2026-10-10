# Sequence learning for early fraud detection (Experiment 2)

**Status: review complete, awaiting your approval. No model trained, no final data generated, nothing merged.**

| File | What it is |
|---|---|
| [`REVIEW_CHECKLIST.md`](REVIEW_CHECKLIST.md) | **Start here.** Git/merge plan and risks, where the ablation is and what is missing, verdicts on the protocol, issues, required fixes, tests, commands, runtime/disk, stop conditions, what to do next. |
| [`PROTOCOL_DRAFT_v1.md`](PROTOCOL_DRAFT_v1.md) | The **current** proposed protocol (evidence-backed changes from v0). Not frozen. |
| [`PROTOCOL_DRAFT.md`](PROTOCOL_DRAFT.md) | v0, kept unchanged for the audit trail. Superseded by v1. |
| [`AUDIT.md`](AUDIT.md) | Phase 1 audit of the generator, features, splits, metrics. Two factual corrections are marked in place; earlier text is in git history. |
| [`review_evidence/`](review_evidence/) | Scripts and raw outputs behind the measured/simulated statements. Not results. |
| [`requirements.txt`](requirements.txt) | Versions used for the measurements. |

Nothing here modifies the production application or any completed experiment. The experiment depends on the v2 generator and
evaluation code from `origin/Keerthan` (see `REVIEW_CHECKLIST.md` §0) and on `experiments/lstm_value_ablation/`, which has not
been available to read.
