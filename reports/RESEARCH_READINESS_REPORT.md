# Research-readiness report

Scope: what the project can and cannot support in a conference paper **today**. It separates confirmed software behaviour, verified synthetic-data findings, missing experiments, unresolved methodological weaknesses, the evidence needed for independent validation, and the claims that can and cannot be made.
Based on the repository at the commit that adds this file (software changes `6c7eea9` and the R-02 change; Experiment 3 `b58fa52`…`1c235ca`), the audit documents in `reports/`, the Experiment 2 documents and the Experiment 3 protocol and results.
Nothing was re-run or overwritten: Experiment 2 has no results, Experiment 3 and all earlier hold-out evidence were only read. The IEEE manuscript has not been started.

## 1. Confirmed software behaviour (verified by tests, probes or browser runs)

| Behaviour | Evidence |
|---|---|
| Scoring pipeline end to end: nine past-only features → LSTM (≥ 10 earlier transactions) → random forest → Fraud Score (0–99.9) → Tree SHAP reasons → alert level; histories persisted and replayed | backend suite 719 passed, 4 skipped, 0 failed after the R-02 change (`reports/FIX_AND_TEST_LOG.md` §7–8); browser runs |
| Input validation, rollback of in-memory history on persistence failure, batch limits, health check, safe PDF names | 41 + follow-up tests; before/after probes (`reports/evidence/`) |
| Behavioral Similarity is withheld (`null`, `insufficient_history`) below 10 earlier transactions | `test_similarity_history.py`; 16 frontend tests; browser run |
| PDF reports are built from the stored transaction; clients cannot alter printed values | `test_report.py` tamper tests |
| Alert levels for the default forest use the validated cut-offs (Low < 4.39 ≤ Medium < 26.23 ≤ Critical); other model sets keep the legacy bands | `docs/R-02-alert-score-decision.md`, `test_alert_policy.py`, frontend tests |
| Model artifacts of the default set are SHA-256-verified before the pickle is loaded | `model_sets.py`, tests |
| Limits: no authentication, TLS or rate limiting; single-process in-memory histories; per-request cost grows with history | register R-01, R-03; `docs/deployment-security-requirements.md` |

Software correctness is **not** scientific validity: none of this says the model detects real fraud.

## 2. Verified synthetic-data findings (all from the project's own generators)

Every number below was either recomputed from committed or saved raw scores or reproduced by an independent script, and is **in-distribution**: the models were developed on the same generator that produced the evaluation data.

| Finding | Where verified | Strength |
|---|---|---|
| The v2 time-split metrics in the committed files are reproduced exactly from the raw scores | `ML_EXPERIMENT_STATUS.md` §3 | exact, but the test period has only 61 fraud rows |
| On fresh hold-out seeds 501–505 the default LSTM → random forest ranks fraud better than the DNN variants and the previous model: PR-AUC 0.464 [0.419, 0.509], recall 0.645 and precision 0.236 at about 10 false alerts per 1,000 legitimate transactions | protocol-hashed `final_holdout_report.json` (read, not re-run) | pre-registered, single use; same generator |
| Experiment 3 (seeds 9301–9305, protocol committed before data): same ordering of the three models (PR-AUC 0.507 / 0.373 / 0.172); the LSTM input adds PR-AUC +0.083 [0.069, 0.096] over the same forest **without** it, both on the nine point features | `experiments/validation_exp3/RESULTS.md`; 51-check recomputation; identical second run | strong within the generator; the no-LSTM forest has **no history aggregates**, so this is not the aggregates-vs-sequence question |
| Behavioral Similarity separates fraud from legitimate rows (ROC-AUC 0.866) but adds nothing measurable beyond the classifier (PR-AUC +0.0011 [−0.0002, +0.0023]) | Experiment 3 | circular by construction (same features as the generator's fraud rules) |
| The forest's score is close to calibrated *on this generator* (slope 1.10); fitted calibrators change the Brier score by 1–4 % | Experiment 3; hold-out reliability table shows under-estimation in the mid-range | generator-specific; not a calibration claim |
| Frozen validation cut-offs generalise across fresh seeds: alert cut-off recall 0.645 / 0.644, precision 0.236 / 0.248 (hold-out / Experiment 3) | `reports/evidence/output/r02_alert_band_impact.json` | consistent; same generator |
| New-customer weakness: with fewer than 10 earlier transactions about 5.9 % of legitimate rows alert at the same cut-offs (README: 59.4 per 1,000) | Experiment 3; hold-out report | measured, not remedied |
| The generator is deterministic across environments | seed 42 regenerated, five hashes matched | exact |

## 3. Missing experiments

| Experiment | Status | What it would answer |
|---|---|---|
| **Aggregates vs sequence** (Experiment 2): tabular models with historical aggregates against LSTM/GRU, first-fraud recall at a fixed FPR | protocol drafts v0/v1/v2; **no data generated, no model trained**; the prior ablation files exist nowhere readable | whether the LSTM is needed at all once aggregates are available — the central claim of the project's design |
| **Warning lead time** | defined in `PROTOCOL_DRAFT_v2.md` §8; not measured | only here could an "early warning" statement be tested |
| **Independent public data** (IEEE-CIS) | compatibility report and protocol draft written; nothing downloaded | behaviour on real labels with a proxy entity (no lead time possible) |
| Generator B (independent synthetic specification with sequential structure) | specified in the Experiment 2 draft; not written | sensitivity to the generator's assumptions |
| New-customer track (late joiners) | generator extension exists; Experiment 3 did not use it | fraud on fewer than 10 earlier transactions, the platform's documented weak spot |
| Calibration and similarity on independent data | measured only on the generator | whether either survives outside it |
| Training-seed variance of the random-forest default | only seed 14 is saved for the forest; the LSTM seed was selected among five | winner's-curse check |
| Explanation faithfulness / usefulness | SHAP additivity is tested; faithfulness and analyst usefulness are not | a claim about "explainable" beyond "SHAP values of the model" |
| Latency, load, drift, adversarial robustness | not evaluated | deployment claims |

## 4. Unresolved methodological weaknesses

1. **One generator, built by the authors.** Fraud archetypes, legitimate behaviour (almost order-free), feature definitions and the hard-coded Indian city list are the authors' assumptions; the first data version was trivially separable (a single feature gave ROC-AUC 1.000). Any win may be a property of the generator.
2. **Repeated use of the same generator family for selection and confirmation.** Fresh seeds protect against overfitting to a draw, not to the generator's assumptions. Several design decisions (features, thresholds, classifier family) were made after earlier looks at the same generator.
3. **Uncertainty is partial.** Bootstrap intervals resample customer components of fixed trained models; training-seed variance (for the forest, a single seed) is not included. The v2 time-split test has 61 fraud rows.
4. **The LSTM's role is unclear.** The project's own documents say it does not warn before a first fraud; first-fraud recall (0.738) exceeds row recall (0.645) for reasons not analysed; on v2 the DNN-only variant detects first frauds at least as well as the LSTM variant. Experiment 3's gain of the LSTM is over a forest *without any history*, not over aggregates.
5. **The Behavioral Similarity Score** is a heuristic with equal weights; 94.5 % of scored rows have a zero-spread feature that the formula replaces by 1; it is circular on this generator and adds nothing beyond the classifier.
6. **No calibration** in serving; the alert levels are cut-offs on a model score. Their evidence (validation FPR targets) is synthetic.
7. **Cold-start** performance is poor (alerts on 5.9 % of legitimate early-history rows) and no remedy is evaluated.
8. **Feature realism.** `is_new_device`, `is_new_location` and `failed_logins_24h` are supplied by the generator; the live application's histories come from the v1 seed while the model was trained on v2.
9. **Multiple analyses.** Experiment 3 reports many intervals; its primary rules were pre-specified, but secondary comparisons are descriptive.
10. **Explanations:** Tree SHAP explains the forest's score, not the fraud mechanism; the LSTM is not explained.
11. **Software:** no authentication; irrelevant to the science but relevant to any deployment claim.

## 5. Evidence required for independent validation

See `experiments/validation_exp3/INDEPENDENT_VALIDATION_STATUS.md` and, for the chosen route, `experiments/independent_ieee_cis/`. In short: a dataset the project did not generate, real or at least independently produced labels, a documented entity or an honestly labelled proxy, chronological splitting with an embargo, features computed from past rows only, thresholds and calibration fitted on a separate period,
a protocol committed before the final period is opened, a single scoring of the final period, paired cluster bootstrap against a no-sequence baseline, and publication of negative results. Status: **none of this evidence exists yet; IEEE-CIS is a candidate whose suitability for histories is uncertain (no customer id) and whose licence has not been read.**

## 6. Claims for a conference paper

### Can be made (with the stated qualifiers)

* The platform design and implementation: per-customer behavioural features, an LSTM-derived score, a random forest, Tree SHAP reasons, persistence, a dashboard; tested software with the verification described above.
* On the project's synthetic generator, with pre-registered evaluation on fresh seeds: the LSTM → random forest design ranks fraud better than the DNN variants and simple baselines, and **the LSTM score adds detectable value over the same forest without any history information** (Experiment 3), at about 10 false alerts per 1,000 legitimate transactions with precision about 0.25.
* The score is a model score, not a probability; calibration on the generator is close but this is not transferred; a Behavioral Similarity heuristic is informative on the generator and adds nothing beyond the classifier.
* Honest limitation reporting: single generator, cold-start weakness, no independent data, partial uncertainty.
* Methodology contributions that are real: leakage-controlled evaluation, protocol-before-data discipline, independent recomputation and a second-run reproducibility check.

### Cannot be made now

* Any statement about real banking data, real fraud, or deployment performance.
* "Early detection", "early warning", "predicts fraud before it happens", "lead time" — none has been measured; the project's own evidence says the LSTM does not warn before a first fraud.
* "The LSTM is better than historical aggregates" (the aggregates baseline has not been run) or "sequence modelling is necessary".
* "Calibrated probabilities", "explainable" in the sense of faithful or useful explanations, or any claim that the similarity score detects fraud independently.
* Generalisation beyond the generator (another data source, another fraud mix, other countries or currencies).
* Superiority of the random forest over other tabular models beyond the pre-registered comparison on this generator.

### What would change the answer

The Experiment 2 result (aggregates vs sequence, first-fraud recall) and a completed independent-data evaluation with its own pre-registered protocol. A paper can be drafted around the platform and the synthetic-data findings with the limitations above, but its central scientific claim (the value of the sequence model) is unsettled until Experiment 2 or the independent evaluation reports.

## 7. Recommended order

1. Owner decisions in `experiments/independent_ieee_cis/DECISIONS_FOR_APPROVAL.md` (D1–D10) — especially access, terms, and whether to proceed with IEEE-CIS despite the proxy entity.
2. IEEE-CIS stage 0 (label-blind audit) → review → entity choice → freeze.
3. In parallel, restore the Experiment 2 ablation files or approve AGG-v2; run Experiment 2 after its Freeze-1.
4. Write the paper only after at least one of 2/3 has reported; until then, restrict the draft to the platform and the synthetic findings.

Documents: Experiment 2 `PROTOCOL_DRAFT_v2.md` (draft; v0 and v1 unchanged), `experiments/independent_ieee_cis/`, `docs/R-02-alert-score-decision.md`.
