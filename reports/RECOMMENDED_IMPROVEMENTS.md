# Recommended improvements

Ordered by evidence and impact, not by effort. Every item points to the register entry that supports it. None of these has been implemented unless the
register says "fixed"; items marked **Decision** need the project owner.

## Priority 1 — decisions that change behaviour

1. **Decide how alerts are defined (R-02) — Decision.** The dashboard's Low/Medium/High/Critical bands are the old DNN's 25/50/80 and are applied to the random
   forest. The model's own validated operating points (alert budget ≈ 10 per 1,000 at score 4.39; critical ≈ 1 per 1,000 at 26.2) are never used to alert.
   Options: (a) adopt those two cut-offs as the Medium/High boundaries and keep Critical for the "critical" policy; (b) keep bands but relabel them as score
   ranges, not risk levels. Either way, add tests and update the README and PDF report wording. Evidence: `/model-info`, `final_holdout_report.json`.
2. **Authentication and abuse protection before any network exposure (R-01) — Decision.** Add an API key or gateway auth, TLS, and rate limiting; the
   un-merged `keerthan` branch has a body-size limiter and per-IP limiter that can be reviewed and cherry-picked once it is clear which branch is the line of record.
3. **What the similarity score returns with no baseline (R-05) — Decision.** For a brand-new customer an ordinary purchase shows 98 % "deviation". Prefer
   `null` (and a visible "not enough history" state) over a number computed against placeholder zeros.
4. **Make PDF reports tamper-resistant (R-04) — Decision.** Look scores up by `transaction_id` on the server instead of printing what the client sends, if reports
   are ever used as records.

## Priority 2 — engineering

5. **Incremental behavioral features (R-03).** Replace the full-history rebuild per request by rolling statistics updated in O(1); keep the current implementation as
   the reference and test equivalence on every customer. Until then, keep the batch cap (F-08) and document the history-length limit (≈6 s at 3,200 transactions).
6. **Frontend tests (R-12).** Add contract tests for the response fields the dashboard reads (`risk_score`, `fraud_probability`, `similarity_pct`, `reasons`) and
   component tests for the error and loading states; show validation errors as readable text instead of raw JSON (R-11).
7. **Reproducible v1 data (R-09) and one feature version (R-08).** Give `generate_synthetic_data.py` a fixed start date; regenerate the v1 feature file with the
   current feature code only if the `production` model set is to remain supported (its model was trained on the old feature scale).
8. **Pin and verify everything that is loaded (R-07, R-10).** Hash-pin `models/saved/` if it stays loadable; pin `keras` exactly; add `pip-audit` and `npm audit`
   to CI; add a migration tool (Alembic) before the schema changes again; use timezone-aware UTC throughout.
9. **Housekeeping (R-10).** Remove `_to_delete/` from tracking; keep large artifacts out of git (consider Git LFS for `.keras`/`.joblib`).

## Priority 3 — science and evaluation (see `ML_EXPERIMENT_STATUS.md`)

10. **Independent evidence.** Evaluate on an independently designed generator and, where permitted, a public dataset, reported separately from synthetic results.
11. **Calibration.** Fit on validation data only; report reliability curves and ECE on a fresh hold-out; then define alert bands from the calibrated scores.
12. **Evaluate the Behavioral Similarity Score** against labels before presenting it as evidence; consider learned feature weights.
13. **First-fraud detection.** The planned protocol in `experiments/sequence_early_detection/` is the route; it needs your approval and the missing ablation files first.
14. **Monitoring.** Log score distributions, alert rates and input ranges per model version (the database already records `model_set` and `model_version`) and alert on drift.

## Not recommended

* Rewriting the architecture, replacing the random forest, or retraining models to improve a metric: the evidence does not call for it, and the hold-outs are single-use.
* Upgrading every dependency: only axios needed a change (F-10); `pip-audit` found nothing in the backend set.
