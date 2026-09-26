# Model evaluation: methodology and results

This document describes how the models are evaluated (Step 4B, corrected methodology). It compares the corrected numbers with the original evaluation and explains why they differ. The numbers come from `backend/models/evaluation/evaluation_report.json`, which is produced by:

```bash
cd backend
python -m app.evaluation.run          # ~11 min on a 2-core CPU
```

`GET /metrics` serves the two headline models from that report. `GET /metrics/report` serves the whole report.

## 1. Old evaluation (before Step 4B)

| Aspect | Old method |
|---|---|
| Split | `train_test_split(test_size=0.2, stratify=y, random_state=42)` over rows, called separately in `lstm_model.py`, `dnn_model.py` and `evaluate.py` |
| Time | train and test interleaved across Jan–Jul 2026; training contained future transactions |
| Customers | all 500 customers in both train and test |
| LSTM windows | a test window shares 9 of 10 rows with a training window 95.6% of the time |
| LSTM → DNN | every DNN training row carried an in-sample LSTM score. The DNN test set was leak-free only because both scripts happened to produce the same split (with another seed, ~80% of test rows would have been leaked). |
| Threshold | fixed at 0.5 on uncalibrated, class-weighted probabilities |
| Metrics | precision, recall, F1, ROC-AUC, confusion matrix; no PR-AUC, baselines or episode metrics |
| Models evaluated | the production models themselves |

## 2. Corrected evaluation

Code: `backend/app/evaluation/` (`split.py`, `windows.py`, `stacking.py`, `metrics.py`, `baselines.py`, `run.py`).

**Primary split: time-based** (`models/evaluation/split_time.json`)

- **Fraud episode:** a customer's fraud transactions, linked while less than 14 days apart. The largest gap between a customer's successive fraud transactions is ~10 days. That gives **60 episodes**, one per fraud customer.
- **Boundaries:** targets between the episodes at 60% and 80% of episode start times. Each target is then moved to the nearby midnight that the fewest episodes straddle.
  - **Train:** before **2026-05-06 00:00**
  - **Validation:** 2026-05-06 to **2026-06-01 00:00**
  - **Test:** from 2026-06-01
- **Straddling episodes:** the 6 that remained are moved whole to the *later* split (3 to validation, 3 to test). No episode crosses a split, and training contains nothing at or after 2026-05-06.

| Split | Transactions | Fraud | Episodes | Period |
|---|---|---|---|---|
| Train | 61,143 | 454 (0.74%) | 32 | 2026-01-12 → 2026-05-05 |
| Validation | 13,866 | 164 (1.18%) | 12 | 2026-05-06 → 2026-05-31 (+ 3 moved episodes from 04-27) |
| Test | 18,904 | 201 (1.06%) | 16 | 2026-06-01 → 2026-07-06 (+ 3 moved episodes from 05-25) |

**Secondary split: customer-grouped** (`split_customer.json`). Customers are split 60/20/20 with seed 42, stratified by whether they have fraud, so test customers are never seen in training. Episodes: 37 / 13 / 12.

**Windows.** The target is each transaction that has at least 10 earlier transactions; its window is the 10 transactions immediately before it, in (timestamp, transaction_id) order. The windows are identical to what production `build_sequences` produces, and a test checks that every window row is strictly earlier than its target.

**Leakage-safe stacking.**

- **Training rows:** they get out-of-fold LSTM scores. Three customer-grouped, fraud-stratified folds are used (`StratifiedGroupKFold`, seed 42). Each fold's LSTM is trained on the other folds and scores only customers it never saw.
- **Validation and test rows:** they are scored by a final LSTM trained on all training rows.
- **DNN:** it trains on the out-of-fold scores (`risk_score` = probability × 100, the same scale production uses).

**Training recipe.** The architectures and hyperparameters are unchanged. Early stopping uses the chronologically last 15% of each model's own training rows, not Keras's random `validation_split`. The validation period is used only for thresholds and comparison.

**Threshold.** It is chosen on validation as the F1-maximizing cut (the midpoint between neighbouring validation scores). The recall-at-FPR operating points are also chosen on validation. Test labels are never used for any choice.

**Reproducibility.** The seed is 42 and TensorFlow op determinism is enabled. The split definitions are saved, and the report records the dataset SHA-256 and library versions. Two independent full runs produced identical reports, apart from the timestamp field. On a different CPU or TensorFlow build, floating-point results can differ slightly.

**Evaluated rows.** Each customer's first 10 transactions have no window, so they are not scored; none of them is fraud.

## 3. Results

### Old vs corrected (headline models, test set)

| Model | Evaluation | PR-AUC | ROC-AUC | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|---|
| LSTM | old (random split) | 0.883 | 0.962 | 0.704 | 0.915 | 0.796 | 0.5 |
| LSTM | **corrected (time)** | **0.829** | **0.947** | **0.829** | **0.871** | **0.850** | 0.998 (validation) |
| DNN (+ LSTM score) | old (random split) | 1.000 | 1.000 | 0.837 | 1.000 | 0.911 | 0.5 |
| DNN (+ LSTM score) | **corrected (time)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.542 (validation) |

Corrected, time split, test period (18,904 transactions, 201 fraud):

| Model | Confusion (TN / FP / FN / TP) | Alerts per 1,000 | Recall @ FPR ≤ 0.1% | Recall @ FPR ≤ 1% |
|---|---|---|---|---|
| LSTM | 18,667 / 36 / 26 / 175 | 11.16 | 0.826 | 0.906 |
| DNN (+ LSTM score) | 18,703 / 0 / 0 / 201 | 10.63 | 1.000 | 1.000 |

### Baselines (same split, same rows, thresholds from validation)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Amount/hour rule (unusual hour AND amount ≥ 188% of avg, cutoff from validation) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Logistic regression (9 features) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| DNN without LSTM `risk_score` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| DNN with LSTM `risk_score` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| LSTM alone | 0.829 | 0.947 | 0.829 | 0.871 | 0.850 |

The secondary customer-grouped split gives the same picture:

- **Rule and single-transaction models:** all at PR-AUC 1.000 and recall 1.000.
- **DNN with vs without the LSTM score:** 1 vs 3 false positives out of 18,448 legitimate transactions.
- **LSTM alone:** PR-AUC 0.913, recall 0.932.

### First fraud of each episode (time split, test period)

| Model | Test episodes | First-fraud transactions | First-fraud recall | Overall fraud recall | Episodes detected | Delay (median) |
|---|---|---|---|---|---|---|
| LSTM | 16 | 16 | **0.000** (0/16) | 0.871 | 16/16 | 1.5 fraud txns, 36.1 h |
| DNN (+ LSTM score) | 16 | 16 | 1.000 | 1.000 | 16/16 | 0 |
| Amount/hour rule | 16 | 16 | 1.000 | 1.000 | 16/16 | 0 |
| Logistic regression | 16 | 16 | 1.000 | 1.000 | 16/16 | 0 |
| DNN without LSTM score | 16 | 16 | 1.000 | 1.000 | 16/16 | 0 |

The customer-grouped split matches: LSTM first-fraud recall is 0/12, and every other model gets 12/12.

## 4. Why the results changed, and what they mean

1. **DNN: the scores stayed perfect, and precision rose from 0.837 to 1.000.** Nothing was manipulated.
   - **No leak removed:** the old perfect ROC-AUC was never caused by split leakage, so removing leakage did not lower it.
   - **Why precision rose:** the old 0.5 threshold was arbitrary. The validation-chosen threshold (0.542) sits in the gap between legitimate and fraud scores.
   - **What 1.000 means:** it measures the synthetic data, not the model. A two-condition rule and logistic regression reach exactly the same numbers, because the generator builds every fraud transaction from the signals the features measure (unusual hour, 3–8× the average amount, failed logins).
2. **LSTM: ranking dropped (PR-AUC 0.883 → 0.829, ROC-AUC 0.962 → 0.947).** The corrected split removes future data from training and removes near-duplicate neighbouring windows from the test set. Precision rose and recall fell because the threshold now comes from validation (0.998, since the class-weighted probabilities are compressed near 1) instead of a fixed 0.5.
3. **The LSTM adds no measurable predictive value.** The DNN without `risk_score` performs identically to the DNN with it on the time split, and within 2 false positives on the customer split. Every single-transaction model beats the LSTM on its own.
4. **The LSTM is not an early-warning model on this data.** It misses the first fraud of every test episode (0/16, and 0/12 on the customer split). It only flags an episode after at least one fraud transaction has happened (median 1.5 fraud transactions, about 36 hours). One reason: the generator labels the "ramp-up" transactions themselves as fraud, so there is no unlabeled precursor period to learn from. Do not describe the LSTM as predicting fraud before it happens.

## 5. Caveats

- **Small sample:** there are only 16 test episodes (12 on the customer split), so one missed first fraud moves first-fraud recall by ~6 percentage points. No confidence intervals are reported yet.
- **Different models:** the evaluation models (`models/evaluation/time_split/`) are trained on the training period only. The production models used by `/predict` are still the original ones, trained with the old random split and in-sample stacking. `/metrics` evaluates the training *recipe*, not those exact weights.
- **Synthetic data:** until it has realistic overlap between legitimate and fraudulent behaviour (or a public benchmark is added), near-perfect scores are expected from any reasonable method and should be presented as such.
