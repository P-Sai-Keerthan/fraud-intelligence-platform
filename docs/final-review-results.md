# Final review — model results summary

> **Step 4D update (8 October 2026).** The default model set is now
> `v2_lstm_rf_seed14`: the same seed-14 LSTM, followed by a **random forest**
> instead of a DNN, chosen by a pre-registered comparison and confirmed on a
> fresh hold-out (`docs/model_selection_report.md`). The figures below describe
> the DNN-based models (previous default and the Step 4C candidates) and remain
> correct for them. For the model that runs by default, use
> `docs/model_selection_report.md` (fresh hold-out: PR-AUC 0.464, recall 0.645,
> 10.23 legitimate alerts per 1,000) and `docs/final-demo-verification.md`.

Every number here is copied from the project's own result files. Nothing is
rounded up and nothing is estimated.

| Source | File |
|---|---|
| Final hold-out (Stage C) | `backend/models/evaluation/v2_holdout/final_holdout_report.json` |
| Production on dataset v1 | `backend/models/evaluation/evaluation_report.json` (shown in the Model Performance tab) |
| Written reports | `docs/step4c3e-stage-c-final-evaluation.md`, `docs/EVALUATION.md` |

## Status in one line

**The application runs the production model. `v2_dnn_lstm` seed 14 passed the
final hold-out and is eligible for a controlled-promotion decision, but it is
NOT deployed.**

## 1. Final hold-out (Stage C)

Fresh synthetic v2 data that was never used for training, for choosing
cut-offs or for choosing the model: 5 datasets, 2,500 customers, 482,292
transactions from customers with at least 10 earlier transactions, of which
2,446 are fraud in 550 fraud episodes. Each model alerts at its own cut-off,
frozen before this data was generated. Brackets are 95% confidence intervals.

| | Production | v2_dnn_lstm seed 14 | v2_dnn_only seed 14 |
|---|---|---|---|
| Recall (share of fraud transactions alerted) | 0.279 [0.248, 0.310] | **0.558** [0.517, 0.600] | 0.419 [0.378, 0.460] |
| Legitimate alerts per 1,000 transactions | 10.60 [10.26, 10.95] | **9.11** [8.41, 9.83] | 10.04 [9.40, 10.74] |
| Fraud transactions caught (of 2,446) | 683 | **1,365** | 1,024 |
| Precision | 0.118 | 0.237 | 0.175 |
| Critical-tier recall | 0.155 | 0.217 | 0.190 |
| First-fraud detection (first fraud of an episode alerted) | 0.388 | 0.605 | 0.690 |
| Episodes detected (at least one alert in the episode) | 0.615 | 0.715 | 0.725 |
| PR-AUC | 0.155 | 0.363 | 0.271 |
| ROC-AUC | 0.717 | 0.938 | 0.895 |
| Alert cut-off (model output, 0–1) | 0.9680 | 0.7929 | 0.8425 |
| Critical cut-off (model output, 0–1) | 0.9951 | 0.9431 | 0.9654 |

### Acceptance gates (fixed before the data existed)

| Gate | v2_dnn_lstm seed 14 | v2_dnn_only seed 14 |
|---|---|---|
| Upper 95% bound of legitimate alerts ≤ 10 per 1,000 | 9.83 — **pass** | 10.74 — **fail** |
| Recall ≥ 0.40 | 0.558 — pass | 0.419 — pass |
| Lower 95% bound of recall above production's recall (0.279) | 0.517 — pass | 0.378 — pass |
| Result | **passed all three** | **failed the alert-budget gate** |

`v2_dnn_only` seed 14 is therefore not eligible. Its point estimate (10.04)
is close to the budget, but the gate is on the upper bound of the interval, and
that is 10.74.

What `v2_dnn_lstm` seed 14 does better and worse than `v2_dnn_only` seed 14:
it catches more fraud transactions overall (recall 0.558 against 0.419) with
fewer false alerts, but it is *weaker* on the first fraud of an episode (0.605
against 0.690). The LSTM needs earlier suspicious transactions to react to.

### New customers (not part of the gates)

Transactions from customers with fewer than 10 earlier transactions: 25,000
transactions, 409 fraud.

| | Production | v2_dnn_lstm seed 14 | v2_dnn_only seed 14 |
|---|---|---|---|
| Legitimate alerts per 1,000 | 75.6 | 25.6 | 50.2 |
| Recall | 0.269 | 0.313 | 0.369 |

Every model raises far more false alerts on new customers than on customers
with history. For seed 14 it is 25.6 per 1,000 against 9.11. No new-customer
policy has been approved, so the 9.11 figure must not be quoted for new
customers. This is an open promotion blocker.

## 2. Production model on dataset v1 (what the Model Performance tab shows)

Time-based evaluation: trained on the earliest transactions, threshold chosen on
the following period, measured on the latest period (18,904 transactions, 201
fraud, 16 fraud episodes). These are evaluation copies of the production
architecture, not the deployed weight files.

| | DNN fraud classifier | LSTM temporal risk model |
|---|---|---|
| Precision | 1.000 | 0.829 |
| Recall | 1.000 | 0.871 |
| F1 | 1.000 | 0.850 |
| PR-AUC | 1.000 | 0.829 |
| ROC-AUC | 1.000 | 0.947 |
| Alerts per 1,000 | 10.63 | 11.16 |
| Confusion matrix (TN / FP / FN / TP) | 18,703 / 0 / 0 / 201 | 18,667 / 36 / 26 / 175 |

**How to present the 1.000 values.** They describe dataset v1, not a perfect
model. In v1 every fraud was generated from the same signals the features
measure (unusual hour, 3–8× the usual amount, failed logins), so a
two-condition rule and a logistic regression also score exactly 1.000
(`docs/EVALUATION.md`). That finding is the reason dataset v2 was built. On the
harder v2 final hold-out the same production model reaches recall 0.279.

## 3. What can and cannot be claimed

Can be said:

- On unseen synthetic v2 data, seed 14 caught about twice as many fraud
  transactions as the production model (1,365 against 683 of 2,446) with
  slightly fewer false alerts (9.11 against 10.60 per 1,000).
- Seed 14 was chosen by a rule written down before the final data was
  generated, and it passed all three pre-registered gates.
- The candidate is integrated, loadable by name for controlled testing, and
  switching back to production is one environment variable.

Must not be said:

- That any model is 100% accurate in the real world.
- That the Fraud Score is a probability. It is an uncalibrated model score.
- That the system is deployed at a bank or tested on real banking data.
- That seed 14 is deployed, or is the model running in the demo.
- That results on synthetic data equal performance on real transactions.
- That seed 14's 9.11 alerts per 1,000 applies to new customers.
- That the live alert levels use the frozen cut-offs. Live scoring uses fixed
  bands (25 / 50 / 80 on the 0–100 Fraud Score) for every model set; the
  Stage C figures describe alerts at Fraud Score ≥ 79.29.
