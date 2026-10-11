# R-02 decision record — what the Fraud Score is, and how it becomes an alert level

Status: **implemented, awaiting owner review.** All numbers below are from synthetic data. Evidence files: `backend/models/evaluation/downstream/final_holdout_report.json`
(protected, read only), `reports/evidence/output/r02_alert_band_impact.json` (produced by `reports/evidence/scripts/r02_alert_band_impact.py`), `experiments/validation_exp3/RESULTS.md`.

## 1. What the model returns

| Question | Answer (verified in the code and the committed reports) |
|---|---|
| Quantity | `predict_proba[:, 1]` of a 200-tree random forest (depth 12, min leaf 5, no class weights) whose inputs are the nine behavioural features and the LSTM Risk Score (`inference_pipeline.py`, `model_sets.py`). It is the mean over trees of the fraction of fraud rows in the leaf the transaction reaches |
| Scaling | multiplied by 100, capped at 99.9 and returned as `fraud_probability` (the field name is historical; the dashboard calls it Fraud Score) |
| Probability? | **No.** It is a model score. Fraud was 0.47–0.49 % of the training and evaluation rows; the forest was trained without reweighting, so low scores are dense and the score range is compressed |
| Calibrated? | Not by construction and not demonstrated. On the fresh hold-out (seeds 501–505, rows with full history) the reliability table shows mid-range scores under-state fraud: scores averaging 0.34 correspond to an observed fraud rate of 0.59, 0.45 → 0.67, 0.55 → 0.85. Overall calibration error looks small (ECE 0.0008) only because 99 % of rows score near 0. Experiment 3 (fresh seeds): calibration slope 1.10; sigmoid/isotonic calibrators lower the Brier score by 1–4 % |
| What /model-info says | `score_semantics.calibrated_probabilities = false`; the wording already says the score "under-states how often the transaction is fraud" in the middle and upper range |

## 2. Current behaviour (before this change)

* `alert_level_from_probability`: Low < 25 ≤ Medium < 50 ≤ High < 80 ≤ Critical, applied to **every** model set, including the random forest. These cut points were chosen for the earlier DNN, whose scores are spread over 0–100.
* The random forest's validated operating points are score cut-offs of **4.39** (alert: false-positive rate within 1 % on the validation period) and **26.23** (critical: within 0.1 %); they were used for every reported metric and recorded as "NOT applied by /predict".
* Dashboard: the verdict banner printed the level; both the Risk Score ring and the Fraud Score ring were tinted by the same 25/50/80 cut points (the Risk Score is the LSTM output, a different quantity with no validated cut points); `/model-info` and the Model Performance tab labelled the bands "legacy fixed bands, not derived from any evaluation of the loaded model".

## 3. Evidence on what each rule does

Fresh synthetic hold-out, seeds 501–505 (487,368 rows, 2,390 fraud), rows with at least 10 earlier transactions (committed report):

| Rule on the forest's score | Fraud found (recall) | Alerts that are fraud (precision) | False alerts per 1,000 legitimate |
|---|---|---|---|
| Legacy "Medium or above" (≥ 25) | 37.9 % (100 − 62.1 % in "Low") | not reported | ≈ 1 (1 − 0.999 of legitimate rows in "Low") |
| Legacy "Critical" (≥ 80) | 2.3 % | not reported | ≈ 0 |
| **Validated alert cut-off (≥ 4.39)** | **64.5 %** | **23.6 %** | **10.2** |
| **Validated critical cut-off (≥ 26.23)** | **36.4 %** | **66.6 %** | **0.9** |

Independent re-check of the same frozen cut-offs on Experiment 3's test seeds (590,877 rows, 2,910 fraud; data not used to choose them): alert cut-off recall 0.644, precision 0.248, 9.6 false alerts per 1,000; critical cut-off 0.383, 0.737, 0.7; legacy ≥ 25: 0.396, 0.715, 0.8; legacy ≥ 80 finds 3.0 % of fraud; 60.4 % of fraud rows show "Low Risk" under the legacy bands, 35.6 % under the validated alert cut-off.

Reading: the legacy "Medium" boundary happens to sit next to the validated *critical* cut-off, so the legacy bands behave like a high-precision, low-recall rule, but they label everything below 25 "Low Risk", which hides about 60 % of fraud that the model's own validated operating point would flag. Dashboard colours and labels therefore did **not** match the thresholds the evidence supports.

## 4. Options considered

| Option | What changes | Verdict |
|---|---|---|
| A. Relabel only | keep 25/50/80, rename the levels ("score band 0–25 …") | does not fix the misleading "Low Risk"; changes every client's wording anyway |
| **B. Apply the validated cut-offs to the default forest** | `alert_level` from the two frozen cut-offs | **chosen**: uses only numbers fixed before the hold-out existed and confirmed twice (hold-out, Experiment 3); no new threshold is invented |
| C. Calibrate the score and keep probability-style bands | add Platt/isotonic to serving | out of scope by instruction; Experiment 3 shows only a small gain; independent data needed first |
| D. A new four-level scheme | invent a third cut-off | no evidence supports one; would be a new, untested threshold |

## 5. Decision implemented (smallest justified correction)

* New `backend/app/alerting.py`. For the default model set `v2_lstm_rf_seed14`: **Low Risk** below 4.39; **Medium Risk** from 4.39 (inclusive); **Critical Risk** from 26.23. **High Risk is not emitted** (no validated third operating point); the field values stay inside the existing four-name vocabulary, so clients keep working.
* Every other model set (`production`, `v2_dnn_lstm`, `v2_dnn_only`, `v2_dnn_lstm_seed14`) keeps the legacy bands exactly (tested against the old function).
* The cut-offs are the constants `RF_FROZEN_CUTOFFS`; start-up refuses to run if they differ from the model manifest's `cutoffs_validation`; a test also checks them against the selection record and the final report.
* `/model-info` now publishes `thresholds.alert_policy` (kind, tiers, levels used, and the measured recall/precision/false-alert rate of the two tiers read from the committed report). The dashboard colours the **Fraud Score** ring by those tiers, colours the **Risk Score** ring in the neutral brand colour (no validated cut points exist for it), words the verdict as "at or above the alert cut-off / critical cut-off", and shows the measured tier performance in Model Performance.
* README limitation text and the register updated. Historical documents (`docs/final-*`, demo scenarios) describe the legacy bands and are left as dated snapshots.

**Naming caveat for the owner.** The 4.39–26.23 tier is labelled "Medium Risk" because it is a *review* tier: of the alerts in that tier alone only about 13 % are fraud (hold-out: 673 of 5,222; Experiment 3: 759 of 6,035). "Critical Risk" alerts are 67–74 % fraud. If you prefer different names (for example "Review" and "Critical"), that is a vocabulary change to every client and test and was deliberately not made here.

## 6. Expected impact

* **More alerts.** About 10 per 1,000 legitimate transactions reach "Medium or above" (before: about 1). In the dashboard the audit's "typical purchase" (score 6.01) is now **Medium Risk** instead of Low Risk. That is the validated 1 % false-positive budget working as designed, not a new error.
* **Fraud found:** from about 38–40 % to 64–65 % of fraud rows at "Medium or above"; "Critical" now catches about 36–38 % (before 2.3–3.0 %).
* **New customers:** with fewer than 10 earlier transactions the model scores neutral inputs high; at these cut-offs 5.9 % of such legitimate rows alert (Experiment 3; README quotes 59.4 per 1,000 on the new-customer hold-out). Expect more "Medium" alerts on new customers. No cold-start policy exists and none was invented.
* **Stored data.** Transactions stored before the change keep the `alert_level` computed with the legacy bands; the database does not record which policy produced a level. History charts and reports can therefore mix semantics for the same model version (register R-16).
* **Reports and batch scoring** use the same level function; the batch summary still lists all four levels (High is 0 for the default model).

## 7. Does this change the scientific protocol? No.

* The evaluation code uses its own frozen tables (`holdout.CUTOFFS`, selection records) and never imports the serving policy (a test asserts that).
* No threshold was re-chosen: the two cut-offs are the values frozen on the validation period before the hold-outs existed. The protected hold-out reports, models and datasets are unchanged (`git diff` shows no change under `backend/models` or `experiments/sequence_early_detection`).
* No calibration was added to serving; the score is still a model score and `calibrated_probabilities` stays false.

## 8. Tests

`backend/tests/test_alert_policy.py` (boundaries including the cut-off itself, High never emitted, legacy function unchanged, only the default forest switches, band texts, cut-offs equal manifest/selection record/final report, evaluation does not import serving policy, API levels consistent, a forced 0.06 score is Medium, `/model-info` policy and evidence, other sets still legacy) and three frontend tests (tier colours, banner wording, legacy fallback). Results are recorded in `reports/FIX_AND_TEST_LOG.md`.

## 9. What remains the owner's decision

1. Accept the Low/Medium/Critical naming and the unused "High Risk", or choose other names.
2. Whether the other model sets should get validated cut-offs (the seed-14 DNN set has frozen ones; they are not applied).
3. Whether a cold-start alert policy is wanted.
4. Whether to record the alert-policy version with each stored transaction.
5. Calibration in serving, only after independent data.
