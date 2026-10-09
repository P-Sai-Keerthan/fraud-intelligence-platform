# Choosing the classifier behind the LSTM (Step 4D)

**Result in one paragraph.** The LSTM was kept unchanged. Behind it, a random
forest replaced the DNN. Four classifiers received identical inputs: the 9
behavioral features and the Risk Score from the same LSTM. They were compared
on five independent development datasets and five training seeds. The comparison
used a selection rule written down before any of them was scored. The random
forest had the highest PR-AUC, and it was the only family that clearly beat the
DNN on recall as well. A fresh hold-out, generated after the selection was
recorded and scored once, confirmed the result: PR-AUC **0.464** against
**0.328** for the DNN with the same LSTM, and recall **0.645** against
**0.548**. Both were measured at a validation-chosen alert budget of about 10
legitimate alerts per 1,000 transactions. All data is synthetic. These numbers
describe the synthetic generator, not real banking traffic. The Fraud Score
remains a model score, not a calibrated probability.

Every number in this document is copied from a file produced by the code. The
files are listed in section 12. Nothing was estimated.

## 1. Problem definition

* **Task:** flag a fraudulent card transaction at the moment it is made. The
  inputs are the transaction itself and the customer's own history.
* **Pipeline:**

  ```
  previous 10 transactions (9 features each) --> LSTM --> Risk Score (temporal risk signal, 0-100)
  9 behavioral features of this transaction + Risk Score --> downstream classifier --> Fraud Score (0-100)
  ```

* **Class balance:** fraud is rare. It is 0.47% of transactions in the v2
  training dataset and 0.50% of the scored development transactions.
  Predicting "legitimate" for everything therefore already scores more than 99%
  accuracy. Accuracy is reported here, but it is never used to choose a model.
* **Primary metric:** PR-AUC (average precision). It measures how well fraud is
  ranked above legitimate transactions, and it does not reward the majority
  class.
* **Operating point:** each model alerts at its own cut-off. The cut-off is
  chosen on the validation period so that at most 1% of legitimate validation
  transactions are alerted (the existing 4C-3B rule, called Policy B). A
  Critical tier is set the same way at 0.1%. Recall, precision and alerts per
  1,000 are measured at that cut-off.

## 2. Why the existing DNN reports about 100% accuracy

The figure comes from the **previous default model set** (`production`). Its
LSTM and DNN were trained on dataset v1. Its evaluation
(`backend/models/evaluation/evaluation_report.json`) shows a perfect test
result for the DNN on the v1 time split:

* 18,703 true negatives, 0 false positives, 0 false negatives and 201 true
  positives;
* accuracy, precision, recall and PR-AUC all **1.000**.

## 3. Leakage audit

Command: `python -m app.evaluation.downstream_audit`. Output:
`backend/models/evaluation/downstream/audit.json`. The checks ran on both
datasets. Production-style windows were used, and the test period was
transactions with at least 10 earlier ones.

| Check | v1 (production data) | v2 (training data of the new model) |
|---|---|---|
| Ground-truth columns in the model frame | none | none (9 metadata columns are loaded separately, analysis only) |
| Model inputs | 9 behavioral features + LSTM risk_score | same |
| Duplicate transaction ids / exact duplicate rows | 0 / 0 | 0 / 0 |
| Training rows at or after the training boundary | 0 | 0 |
| Fraud episodes split across train/validation/test | 0 | 0 |
| Customers shared between customer-split partitions | 0 | 0 |
| Test feature vectors that also occur in training | 0 of 18,904 | 53 of 21,142 (none of them fraud) |
| LSTM windows strictly before their target | checked by `assert_past_only` in every training run | same |
| Scaler fitted on | training rows only (`stacking.fit_dnn`) | same |
| Threshold chosen on | validation period only | same |

**No leakage was found.** Step 4B had already removed the one historical leak
(in-sample LSTM scores in the old random-split evaluation). Removing it did
not lower the DNN's v1 score (`docs/EVALUATION.md`).

**What does explain 100%**, measured on the v1 test period (201 fraud, 18,703
legitimate):

| Evidence | Value |
|---|---|
| Accuracy of predicting "legitimate" for every transaction | **0.9894** |
| One feature alone, `hour_is_unusual = 1` | precision **1.000**, recall **1.000** |
| One feature alone, `amount_pct_of_avg ≥ 310.14` | precision **1.000**, recall **1.000** |
| Logistic regression, two-condition amount/hour rule, DNN without the LSTM score | all PR-AUC **1.000** (`evaluation_report.json`) |
| Fraud transactions whose feature vector also occurs for a legitimate transaction | 0 of 201 |

The same checks on v2, which the generator was built to make harder:

* the best single feature (`hour_is_unusual`) reaches only ROC-AUC 0.731;
* no single feature reaches an F1 above 0.168;
* "always legitimate" scores 0.9971 accuracy on the v2 test period.

**Finding:**

* **A + E.** v1 is a synthetic dataset in which a single feature separates
  fraud perfectly. Any reasonable model gets 100% on it.
* **C + D.** Accuracy is dominated by the 99% legitimate majority, so it
  cannot tell models apart. On the development data every candidate's accuracy
  lies between 0.9882 and 0.9887, while PR-AUC ranges from 0.293 to 0.471. On
  the fresh hold-out, the previous default has accuracy 0.9856 while catching
  only 25.5% of fraud.
* **B (leakage):** not found.

The DNN is therefore not "too good". The v1 evaluation could not measure it.
The comparison below uses dataset v2 and fraud-specific metrics only.

## 4. Candidate models

All candidates take the same 10 inputs: the 9 features plus the out-of-fold
LSTM `risk_score`. They are scaled with the same training mean and standard
deviation.

| Family | Settings tried | Setting chosen (validation PR-AUC, mean of 5 seeds) |
|---|---|---|
| DNN (incumbent) | none (existing recipe and saved models) | (0.240) |
| Logistic regression | C ∈ {0.01, 0.1, 1, 10} × class_weight ∈ {none, balanced} | C = 10, no class weight (0.308) |
| Random forest | min_samples_leaf ∈ {1, 5, 20} × max_depth ∈ {none, 12} × class_weight ∈ {none, balanced_subsample}; 200 trees | min_samples_leaf = 5, max_depth = 12, no class weight (0.388) |
| Histogram gradient boosting | learning_rate ∈ {0.05, 0.1} × max_leaf_nodes ∈ {15, 31} × max_iter ∈ {100, 300} × class_weight ∈ {none, balanced} | 0.05, 15 leaves, 100 iterations, no class weight (0.331) |
| XGBoost | not evaluated: not installed in the project environment; no new large dependency was added | |

* **How settings were chosen:** only on the validation period of the training
  dataset (16,315 transactions, 119 fraud). Source:
  `backend/models/evaluation/downstream/tuning.json`.
* **DNN validation PR-AUC:** 0.243, 0.217, 0.195, 0.283 and 0.260 for training
  seeds 11–15 (mean 0.240).

## 5. Evaluation methodology

* **Pre-registration.** The rule was written in
  `docs/step4d-downstream-selection-protocol.md` (SHA-256 `46dfbc09…`) before
  any candidate was trained. The selection record repeats that checksum.
* **LSTM held fixed.** Five LSTMs were already trained, for seeds 11–15
  (`models/candidates_multiseed/v2/seed_<s>/dnn_lstm/`). For each of them, the
  out-of-fold Risk Scores of the 58,840 training rows (291 fraud) were
  recomputed with the unchanged recipe. The recomputation was accepted only if
  both of these held bit for bit:
  * the final LSTM reproduced the saved LSTM weights;
  * a DNN refit on the recomputed scores reproduced the saved DNN weights.

  Both held for all five seeds (`reproduction.json`). Every candidate trained
  for seed *s* therefore sees exactly the inputs the saved DNN of seed *s* saw.
* **Out-of-fold stacking.** Each training row's Risk Score comes from an LSTM
  that never saw that customer (3 customer-grouped folds). The validation and
  test rows are scored by the final LSTM.
* **Splits.** The training dataset (seed 42) uses the saved out-of-time split:
  * training: before 2026-05-03;
  * validation: 2026-05-03 to 2026-06-01;
  * test period: never used.

  Rings and households are never split.
* **Development data:** datasets 301–305 from the unchanged generator 2.0.1,
  with 500 customers each. Pooled, they hold 489,898 scored transactions:
  2,430 fraud in 550 episodes, from 2,500 customers.
* **Fresh hold-out:**
  * **Main:** datasets 501–505 (487,368 scored transactions, 2,390 fraud, 550
    episodes).
  * **New-customer:** datasets 511–515, with 20% late-joining customers and 30
    new-customer fraud episodes per dataset.
  * **When it was made:** generated only after the selection record had been
    written (the code refuses otherwise), then scored once.

  The earlier hold-outs (101–105, 201–205, 401–405, 411–415) were not used.
  Their results had already been seen.
* **Scoring.** Inputs were built exactly as `/predict` builds them:
  * LSTM over the 10 previous transactions;
  * scaling and a ±6 clip;
  * cold-start imputation;
  * scores capped at 0.999.

  For the DNN, the inputs were checked to equal `holdout.score_frame` bit for
  bit on development dataset 301. The recomputed seed-14 DNN cut-offs equal
  the frozen Stage B values (0.79287 / 0.94308).
* **Population:** customers with at least 10 earlier transactions. For fewer,
  see section 11.
* **Uncertainty.**
  * Within one model: 95% intervals from a group bootstrap that resamples whole
    customer groups within each dataset (2,000 resamples).
  * Between families: the difference of the means over training seeds, with an
    interval that includes both data and training-seed variation.
  * For the single deployed artifact on the hold-out: a paired group bootstrap.

## 6. Metrics on the development data (selection)

Mean over training seeds 11–15 (± standard deviation across seeds). Each model
is at its own Policy B cut-off. Source: `development_report.json`.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Legit alerts/1000 | Accuracy |
|---|---|---|---|---|---|---|---|
| DNN (incumbent) | 0.293 ± 0.043 | 0.935 | 0.214 | 0.515 | 0.302 | 9.40 | 0.9882 |
| Logistic regression | 0.381 ± 0.010 | 0.936 | 0.223 | 0.543 | 0.316 | 9.40 | 0.9883 |
| **Random forest** | **0.471 ± 0.015** | **0.943** | **0.250** | **0.638** | **0.359** | 9.51 | 0.9887 |
| Hist. gradient boosting | 0.388 ± 0.011 | 0.936 | 0.235 | 0.565 | 0.332 | 9.13 | 0.9887 |

| Model | Critical-tier recall | First fraud of an episode detected | Episodes detected |
|---|---|---|---|
| DNN (incumbent) | 0.198 | 0.530 | 0.685 |
| Logistic regression | 0.274 | 0.681 | 0.745 |
| **Random forest** | **0.343** | **0.743** | **0.800** |
| Hist. gradient boosting | 0.280 | 0.640 | 0.761 |

Confusion matrices at Policy B, training seed 14, development data pooled:

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| DNN | 483,093 | 4,375 | 1,080 | 1,350 |
| Logistic regression | 482,951 | 4,517 | 1,038 | 1,392 |
| Random forest | 482,810 | 4,658 | 851 | 1,579 |
| Hist. gradient boosting | 483,199 | 4,269 | 1,048 | 1,382 |

Each challenger minus the DNN: difference of means, with a 95% interval
including data and training-seed variation:

| Challenger − DNN | PR-AUC | Recall | Legit alerts/1000 | Critical recall | First-fraud detection |
|---|---|---|---|---|---|
| Logistic regression | +0.088 [+0.051, +0.125] | +0.028 [−0.016, +0.078] | −0.00 [−0.66, +0.55] | +0.077 [+0.051, +0.106] | +0.151 [+0.104, +0.202] |
| Random forest | +0.178 [+0.139, +0.218] | +0.123 [+0.079, +0.176] | +0.11 [−0.67, +0.76] | +0.145 [+0.107, +0.185] | +0.213 [+0.169, +0.265] |
| Hist. gradient boosting | +0.095 [+0.056, +0.133] | +0.050 [+0.005, +0.103] | −0.27 [−1.15, +0.58] | +0.082 [+0.053, +0.115] | +0.110 [+0.060, +0.164] |

## 7. Seed comparison

PR-AUC on the pooled development data, per training seed:

| Training seed | DNN | Logistic regression | Random forest | Hist. gradient boosting |
|---|---|---|---|---|
| 11 | 0.295 | 0.379 | 0.450 | 0.394 |
| 12 | 0.296 | 0.369 | 0.484 | 0.374 |
| 13 | 0.233 | 0.377 | 0.465 | 0.383 |
| 14 | 0.355 | 0.394 | 0.487 | 0.402 |
| 15 | 0.287 | 0.385 | 0.471 | 0.388 |
| min–max | 0.233–0.355 | 0.369–0.394 | 0.450–0.487 | 0.374–0.402 |

* **Recall range across seeds:** DNN 0.428–0.556; logistic regression
  0.520–0.573; random forest 0.624–0.650; gradient boosting 0.541–0.574.
* **Legitimate alerts per 1,000 across seeds:** DNN 8.81–10.65; random forest
  8.97–10.15.
* **Random forest is stable:**
  * its worst seed (0.450) beats the DNN's best seed (0.355);
  * it beats the DNN for 5 of 5 seeds;
  * it beats the DNN on 5 of 5 development datasets: 0.486 / 0.419 / 0.481 /
    0.507 / 0.468 against 0.315 / 0.237 / 0.310 / 0.314 / 0.295.
* **The DNN is the least stable family** (standard deviation of PR-AUC 0.043,
  against 0.010–0.015 for the other families).

## 8. Calibration

The score is read as if it were a probability, then compared with what
actually happened.

| Model (development data, mean over seeds) | Brier score | Log loss | Expected calibration error (10 bins) |
|---|---|---|---|
| DNN | 0.0718 | 0.258 | 0.189 |
| Logistic regression | 0.0038 | 0.018 | 0.0006 |
| Random forest | 0.0035 | 0.017 | 0.0008 |
| Hist. gradient boosting | 0.0038 | 0.019 | 0.0013 |

For reference, the Brier score of always predicting the fraud rate is 0.0049.

* **The DNN** is trained with class weights, so its scores are far too high:
  the median legitimate transaction scores 0.126.
* **The random forest's** error looks small only because 99% of transactions
  score below 0.1, where score and fraud rate agree. Its reliability on the
  fresh hold-out is in the table below.

| Fraud Score band | Transactions | Mean score | Observed fraud rate |
|---|---|---|---|
| 0–10 | 483,700 | 0.2% | 0.2% |
| 10–20 | 1,760 | 14.4% | 16.1% |
| 20–30 | 840 | 24.3% | 30.6% |
| 30–40 | 362 | 34.3% | 58.6% |
| 40–50 | 281 | 44.8% | 67.3% |
| 50–60 | 181 | 54.9% | 84.5% |
| 60–70 | 104 | 64.6% | 89.4% |
| 70–80 | 85 | 74.5% | 95.3% |
| 80–90 | 51 | 84.3% | 100% |
| 90–100 | 4 | 91.3% | 100% |

* **Above 30 the score clearly understates the fraud rate.** For example,
  scores of 30–40 are fraud 58.6% of the time. This is a ranking score, not a
  probability.
* **Post-hoc recalibration was tested** (`posthoc_calibration_seed_14`). A
  sigmoid and an isotonic map were fitted on the validation period and
  evaluated on the development data. Brier scores were 0.00340 and 0.00339,
  and ECE 0.0026 for both. They were **not** applied:
  * the bands they would change are the application's fixed alert bands
    (section 11);
  * calibration measured on synthetic v2 data would not carry over to the
    application's customers (v1 seed data) or to real data.
* **Decision (pre-registered):** the output stays the **Fraud Score (a model
  score, not a calibrated probability)**. The API field keeps its historical
  name, `fraud_probability`, and `/model-info` reports
  `calibrated_probabilities: false`.

## 9. SHAP verification

| Model | SHAP method | Check |
|---|---|---|
| Random forest, gradient boosting | `shap.TreeExplainer`, interventional, model output = predicted score, 200 training rows as background (exact Tree SHAP) | contributions + expected value = the forest's own output, to 1e-6 (`tests/test_rf_model_set.py`, `tests/test_downstream_selection.py`) |
| Logistic regression | `shap.LinearExplainer` (exact, log-odds) | additivity to 1e-6 |
| DNN (previous default) | `shap.GradientExplainer` (sampled expected gradients) | unchanged |

The application builds the explainer from the loaded model object
(`app/models/downstream_classifier.py`, `app/models/shap_explainer.py`). The
chain is Live Scan → `/predict` → random forest → TreeExplainer on the same
random forest → the reasons in the UI and the PDF. A test checks that the
explained model is the scoring model, and that the values add up to its score.
The previous DNN's SHAP values are not used anywhere for the new default.

## 10. Final model selection

* **Rule outcome on the development data (E1–E7).** All three challengers
  passed every condition:
  * a clear PR-AUC gain over the DNN;
  * recall, alert burden and Critical-tier recall not materially worse;
  * mean recall ≥ 0.40;
  * stable over 5/5 seeds and 5/5 datasets;
  * bit-identical refits.
* **Winner:** the random forest, with the highest mean PR-AUC (0.471 against
  0.388 and 0.381). The gap exceeds the 0.005 tie margin, so calibration and
  cost were not needed to decide. Selection record:
  `backend/models/evaluation/downstream/selection_record.json`.
* **Deployed artifact:**
  * the seed-14 LSTM, copied byte for byte and unchanged;
  * a random forest with min_samples_leaf 5, max_depth 12, 200 trees and no
    class weight, trained with seed 14 on the seed-14 out-of-fold Risk Scores;
  * the training seed was fixed in Stage B of 4C-3E.6, before this protocol.
* **Cut-offs (validation):** Policy B 0.0439, Critical 0.2623.

**Confirmation on the fresh hold-out 501–505, scored once:**

| Model | PR-AUC | ROC-AUC | Precision | Recall | Legit alerts/1000 | First fraud detected | Episodes detected | Accuracy |
|---|---|---|---|---|---|---|---|---|
| **LSTM + random forest (selected)** | **0.464** [0.419, 0.509] | 0.939 | 0.236 | **0.645** [0.608, 0.683] | 10.23 [9.56, 10.91] | **0.738** | **0.798** | 0.9880 |
| LSTM + DNN (same LSTM, seed 14) | 0.328 [0.286, 0.371] | 0.929 | 0.217 | 0.548 [0.505, 0.588] | 9.68 [8.96, 10.40] | 0.595 | 0.702 | 0.9881 |
| Previous default (v1 LSTM + DNN) | 0.133 [0.111, 0.157] | 0.691 | 0.104 | 0.255 [0.228, 0.283] | 10.71 [10.35, 11.06] | 0.384 | 0.585 | 0.9856 |

| Confusion at Policy B | TN | FP | FN | TP |
|---|---|---|---|---|
| Selected | 479,993 | 4,985 | 848 | 1,542 |
| DNN, seed 14 | 480,258 | 4,720 | 1,081 | 1,309 |
| Previous default | 479,758 | 5,220 | 1,781 | 609 |

| Gate (pre-registered) | Result |
|---|---|
| C1: PR-AUC above the seed-14 DNN | +0.136 [+0.116, +0.156]: **holds** |
| C2: recall ≥ 0.40 | 0.645: **holds** |
| C3: legitimate alerts at most +1.0 per 1,000 above the DNN | +0.54 [+0.27, +0.81]: **holds** |

**Outcome:** confirmed. The random forest replaces the DNN behind the unchanged
LSTM (`final_holdout_report.json`).

* **The trade-off, stated plainly:** the random forest raises about 0.5 more
  legitimate alerts per 1,000 than the DNN. The interval excludes zero, but the
  increase is inside the pre-registered limit. In return it catches about 10
  more fraud transactions in every 100.
* **By fraud type on the hold-out** (recall, selected against DNN):

  | Fraud type | Selected | DNN |
  |---|---|---|
  | Card-testing / cash-out | 0.849 | 0.607 |
  | Account takeover | 0.768 | 0.722 |
  | Device takeover | 0.512 | 0.370 |
  | High-value single | 0.795 | 0.718 |
  | Stolen card online | 0.418 | 0.385 |
  | Ring | 0.967 | 0.906 |
  | "Normal-looking" fraud | 0.081 | 0.088 |

  Fraud that the generator makes look normal is still missed by both.
* **Hold-out per dataset (PR-AUC):** 0.501 / 0.478 / 0.467 / 0.372 / 0.507,
  against 0.372 / 0.332 / 0.370 / 0.236 / 0.349 for the DNN.
* **New-customer hold-out, customers with full history (511–515):** PR-AUC
  0.490 against 0.365, and recall 0.638 against 0.533, at 9.50 against 8.78
  legitimate alerts per 1,000.

**Computational cost** (classifier only, this machine's CPU, seed 14):

| | DNN | Logistic regression | Random forest | Hist. gradient boosting |
|---|---|---|---|---|
| One transaction (median, as `/predict` calls it) | 84.1 ms | 0.07 ms | 57.4 ms | 1.9 ms |
| Batch of 4,096 rows | 3.8 ms | 0.5 ms | 86.4 ms | 20.3 ms |
| Model size | 86.8 KB | 0.9 KB | 3.57 MB | 190.7 KB |

The LSTM, which runs first, is the same for all four. The random forest's
single-row time is mostly scikit-learn's per-call overhead. At this scale it is
not a constraint.

## 11. Limitations

1. **All data is synthetic** (generator v2.0.1). The results describe how well
   each model learns this generator. They do not establish performance on real
   banking traffic, and no claim of real-world accuracy is made.
2. **Customers with fewer than 10 earlier transactions** (the LSTM is not run):
   * on the fresh new-customer data (511–515), the random forest raised
     **59.4** legitimate alerts per 1,000, against **25.0** for the DNN and
     73.8 for the previous default;
   * in return its recall there is 0.423, against 0.313;
   * the cold-start behaviour is worse than the DNN's on false alerts. It is an
     open issue, and no cold-start policy has been approved.
3. **The alert bands in the app (25 / 50 / 80) were not derived for this
   model.** The random forest's scores are much lower than the DNN's (a
   median legitimate score of 0.06 against 12.6). Its validated alert cut-off
   is a score of 4.39, and its Critical cut-off is 26.2. Under the fixed bands:
   * 62.1% of hold-out fraud transactions fall in "Low Risk";
   * 99.9% of legitimate transactions fall in "Low Risk".

   Every metric in this report is measured at the validation cut-offs, not at
   the bands. Changing the bands is an owner's decision and was not part of
   this protocol.
4. **The Fraud Score is not a calibrated probability** (section 8).
5. **Mixed data.** The application's live customer histories are the v1 seed
   data, while the deployed model was trained on v2 data.
6. **No pre-fraud claim.** The LSTM produces a temporal risk signal. Its own
   first-fraud recall on v1 is 0/16 (`docs/EVALUATION.md`). The first-fraud
   figures above are for the whole pipeline, and they count a first fraud as
   detected when that transaction itself is alerted. Nothing here shows
   prediction before fraud occurs.
7. **Hyperparameter grids were small** and identical in spirit for all
   families. The DNN was not re-tuned; it kept its existing recipe. XGBoost
   was not evaluated.
8. **The confirmation stage ran three times.** The first run stopped with an
   import error, because code was edited while it ran. The second stopped
   while computing score distributions for an empty class. Neither wrote or
   displayed any result. The third run is the one reported. The data, the
   artifact (same SHA-256) and the rule were identical in all three.

## 12. Reproduction

From `backend/`, with the project's virtual environment. The v2 data is not
committed; the generator reproduces it byte for byte.

```bash
python ../data/v2/generate.py                                   # training dataset v2 (seed 42)
python -m app.evaluation.downstream_audit                       # section 3 -> audit.json
python -m app.evaluation.downstream data                        # development datasets 301-305
python -m app.evaluation.downstream reproduce                   # out-of-fold LSTM scores, bit-for-bit check (~40 min)
python -m app.evaluation.downstream tune                        # validation-only settings -> tuning.json
python -m app.evaluation.downstream develop                     # development comparison + selection_record.json
python -m app.evaluation.downstream confirm                     # fresh hold-out 501-505 / 511-515, scored once
python -m app.evaluation.downstream_sanity                      # hand-built scenarios -> sanity_scenarios.json
python -m pytest -q                                             # full test suite
```

* `confirm` and `develop` refuse to overwrite the order of events: no record
  means no hold-out, and an existing hold-out means no new record. To re-run
  them, start from a clean copy of `models/evaluation/downstream/` and
  `data/v2_holdout/final_4d/`.
* Results were produced with Python 3.13.13, TensorFlow 2.21.0, scikit-learn
  1.8.0, numpy 2.4.4 and pandas 3.0.2. On another CPU or library build,
  floating-point results can differ slightly.

Files (all under `backend/models/evaluation/downstream/` unless noted):

| File | Contents |
|---|---|
| `audit.json` | section 3 |
| `reproduction.json` | per seed: LSTM and DNN weight hashes reproduced |
| `tuning.json` | every setting × seed, validation PR-AUC, chosen settings |
| `development_report.json` | sections 6–8 and 10: every model × seed, intervals, eligibility, calibration, score distributions, cost |
| `selection_record.json` | the decision, written before the hold-out existed |
| `final_holdout_report.json` | the fresh hold-out, new-customer hold-out, gates, calibration, distributions, fraud types |
| `sanity_scenarios.json` | the hand-built transactions in `docs/final-demo-verification.md` |
| `backend/models/candidates_downstream/v2/seed_14/lstm_random_forest/` | the deployed artifact and its manifest |
