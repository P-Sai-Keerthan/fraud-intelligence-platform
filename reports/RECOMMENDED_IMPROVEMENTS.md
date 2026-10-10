# Recommended improvements

Ordered by evidence and impact, not by effort. Every item points to the register entry that supports it. None of these has been implemented unless the
register says "fixed" (follow-up status shown in brackets); items marked **Decision** need the project owner.

## Priority 1 — decisions that change behaviour

1. **Decide how alerts are defined (R-02) — Decision.** The dashboard's Low/Medium/High/Critical bands are the old DNN's 25/50/80 and are applied to the random
   forest. The model's own validated operating points (alert budget ≈ 10 per 1,000 at score 4.39; critical ≈ 1 per 1,000 at 26.2) are never used to alert.
   Options: (a) adopt those two cut-offs as the Medium/High boundaries and keep Critical for the "critical" policy; (b) keep bands but relabel them as score
   ranges, not risk levels. Either way, add tests and update the README and PDF report wording. Evidence: `/model-info`, `final_holdout_report.json`.
2. **Authentication and abuse protection before any network exposure (R-01) — Decision.** The requirements and three options are written in `docs/deployment-security-requirements.md` (gateway SSO, bearer tokens, API key — the last unsuitable for the dashboard). Safe defaults already added: CORS limits, `nosniff`, `no-store`, a declared-length guard, `API_DOCS=off`. Still needed: authentication, role and object-level rules, TLS, rate limits, a single worker or shared state, retention. Add an API key or gateway auth, TLS, and rate limiting; the
   un-merged `keerthan` branch has a body-size limiter and per-IP limiter that can be reviewed and cherry-picked once it is clear which branch is the line of record.
3. **[Done] What the similarity score returns with no baseline (R-05).** Implemented: `null` plus `similarity_status = "insufficient_history"` below 10 earlier transactions, shown as "Not enough history". *Open question for the owner:* is 10 the right threshold for a score that, unlike the LSTM, only needs a mean and spread? (Experiment 3's ROC-AUC for the score is lower at 10–29 earlier transactions than at 100 or more, but only 83 fraud rows fall in the first bin and the intervals nearly touch, so the data do not settle it; see `ML_EXPERIMENT_STATUS.md`.)
4. **[Done] Make PDF reports tamper-resistant (R-04).** Implemented: the report is built from the stored transaction; the client sends only an id. *Remaining:* decide who may fetch a given customer's report (object-level authorisation, item 2) and how long reports and rows are retained.

## Priority 2 — engineering

5. **Incremental behavioral features (R-03).** Replace the full-history rebuild per request by rolling statistics updated in O(1); keep the current implementation as
   the reference and test equivalence on every customer. Until then, keep the batch cap (F-08) and document the history-length limit (≈6 s at 3,200 transactions).
6. **Frontend tests (R-12) [partly done].** 13 tests now cover the similarity states, the error formatter and the PDF request; readable validation errors are done (R-11). Still to add: tests for the batch table, the history timeline, loading states and a browser-level flow for a new customer (the dashboard has no way to create one).
7. **Reproducible v1 data (R-09) [done] and one feature version (R-08).** `generate_synthetic_data.py` now takes a fixed start date and seed; regenerate the v1 feature file with the
   current feature code only if the `production` model set is to remain supported (its model was trained on the old feature scale).
8. **Pin and verify everything that is loaded (R-07, R-10).** Hash-pin `models/saved/` if it stays loadable; pin `keras` exactly; add `pip-audit` and `npm audit`
   to CI; add a migration tool (Alembic) before the schema changes again; use timezone-aware UTC throughout.
9. **Housekeeping (R-10).** Remove `_to_delete/` from tracking; keep large artifacts out of git (consider Git LFS for `.keras`/`.joblib`).

## Priority 3 — science and evaluation (see `ML_EXPERIMENT_STATUS.md`)

10. **Independent evidence — the main gap.** Evaluate on an independently designed generator and, where permitted, real or public data, reported separately from synthetic results. Requirements, leakage risks and the final-evaluation procedure are in `experiments/validation_exp3/INDEPENDENT_VALIDATION_STATUS.md`; no such data exists in the project today and none was downloaded.
11. **Calibration [measured, not adopted].** Experiment 3 measured it (see `ML_EXPERIMENT_STATUS.md`): on this generator the forest score is already close to calibrated and fitted calibrators change Brier and log loss only slightly. *Decision:* whether to add a validation-fitted calibrator to serving and call the number a probability. Do not do so on this evidence alone, because the base rate and customer mix come from the same generator; re-check on independent data first.
12. **Behavioral Similarity Score [evaluated].** Experiment 3: it separates fraud from legitimate rows on the generator's own fraud types but adds no measurable information beyond the classifier, so present it as an explanation aid ("how unusual is this purchase for this customer"), not as a second detector. A learned weighting would need its own protocol (R-14).
13. **First-fraud detection.** The planned protocol in `experiments/sequence_early_detection/` is the route; it needs your approval and the missing ablation files first.
14. **Monitoring.** Log score distributions, alert rates and input ranges per model version (the database already records `model_set` and `model_version`) and alert on drift.

## Not recommended

* Rewriting the architecture, replacing the random forest, or retraining models to improve a metric: the evidence does not call for it, and the hold-outs are single-use.
* Upgrading every dependency: only axios needed a change (F-10); `pip-audit` found nothing in the backend set.

## Decisions waiting for the owner (collected)

1. Alert bands (item 1). 2. Authentication option, role model, object-level rule, rate-limit budgets, hosting model (single worker vs shared state), retention (item 2). 3. The history threshold for the similarity score (item 3).
4. Whether to add calibration to serving (item 11). 5. Which independent data to pursue (item 10). 6. Housekeeping: `_to_delete/`, pin for the `production` model set (R-07, R-10).
7. Approval and missing ablation files for the first-fraud experiment (item 13).
