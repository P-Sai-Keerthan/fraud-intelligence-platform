# Step 4D: pre-registered protocol for choosing the downstream classifier

Protocol version: `4D v1`. Written on 8 October 2026 at commit `33565e3`,
**before** any candidate in this document was trained or scored. The rules
below are also fixed in code (`backend/app/evaluation/downstream.py`); the
selection record states the protocol version it applied.

## 1. Question

The application scores a transaction in two stages:

```
previous 10 transactions --> LSTM --> Risk Score (temporal risk signal)
9 behavioral features + Risk Score --> downstream classifier --> Fraud Score
```

The LSTM stays. This protocol decides which **downstream classifier** should
consume the 9 behavioral features plus the LSTM Risk Score: the current DNN, or
a logistic regression, a random forest or a histogram gradient-boosting model.
The answer may be "keep the DNN". Accuracy is **not** a selection metric.

## 2. What is held fixed

| Item | Fixed value |
|---|---|
| LSTM | Architecture, window (10 previous transactions), 9 input features and training recipe unchanged (`app/models/lstm_model.py`, `app/evaluation/stacking.py`). For training seed *s* the LSTM is exactly the one already trained for `v2_dnn_lstm` seed *s* (`models/candidates_multiseed/v2/seed_<s>/dnn_lstm/`). The out-of-fold Risk Scores of the training rows are recomputed with the unchanged recipe; the recomputation is accepted only if retraining the DNN on them reproduces the saved DNN weights bit for bit (SHA-256 of the weights). |
| Classifier inputs | The 9 behavioral features + `risk_score`, in the application's order. No ground-truth column (`assert_no_ground_truth`). |
| Training rows | Training dataset (generator 2.0.1, seed 42), training period, transactions with at least 10 earlier ones. Training rows carry **out-of-fold** Risk Scores (customer-grouped folds), as for the DNN. |
| Input scaling | Mean and standard deviation of the training rows. The non-DNN candidates are trained on the scaled inputs clipped to ±6, which is exactly what the application gives them when scoring. The DNN keeps its original recipe (trained unclipped, scored clipped). |
| Cold start | As in the application: fewer than 10 earlier transactions → Risk Score set to the training mean; first transaction → baseline-relative features set to the training mean. |
| Cut-offs | Per model, on the validation period of the training dataset only, by the existing 4C-3B rule: Policy B = lowest score with validation false-positive rate ≤ 1%, Critical = ≤ 0.1%. Never re-tuned on development or hold-out data. |

## 3. Candidates and hyperparameters

| Family | Settings tried (grid) | Fixed |
|---|---|---|
| DNN (incumbent) | none: the existing recipe and the saved seed-*s* models | `dnn_model.py` architecture, class-weighted |
| Logistic regression | C ∈ {0.01, 0.1, 1, 10}; class_weight ∈ {none, balanced} | lbfgs, max_iter 2000 |
| Random forest | min_samples_leaf ∈ {1, 5, 20}; max_depth ∈ {none, 12}; class_weight ∈ {none, balanced_subsample} | 200 trees, max_features sqrt |
| Histogram gradient boosting | learning_rate ∈ {0.05, 0.1}; max_leaf_nodes ∈ {15, 31}; max_iter ∈ {100, 300}; class_weight ∈ {none, balanced} | no early stopping, other scikit-learn defaults |
| XGBoost | **not evaluated**: not installed in the project environment, and installing a new large dependency is out of scope | |

**Choosing a family's setting:** the setting with the highest mean PR-AUC on
the validation period of the training dataset, over training seeds 11–15. Ties
go to the earlier setting in the order above. Development and hold-out data
are never used for this.

## 4. Data and what it may be used for

| Data | Seeds | Use |
|---|---|---|
| Training dataset, training period | 42 | Fitting |
| Training dataset, validation period | 42 | Hyperparameter setting; cut-offs |
| **Development datasets** | **301–305** | **Choosing the family (this protocol)** |
| Fresh final hold-out | **501–505** (default generator, 500 customers) | Confirmation only; generated **after** the selection record is written, scored once |
| Fresh new-customer hold-out | **511–515** (20% late joiners, 30 new-customer episodes) | Confirmation only, same conditions |
| Earlier hold-outs | 101–105, 201–205, 401–405, 411–415 | Not used: their results have been seen |

The fresh hold-out is generated with the unchanged generator (version 2.0.1);
no label or row is altered. The code refuses to generate or score it before the
selection record exists.

## 5. Metrics

Population: transactions with at least 10 earlier transactions (primary), the
five development datasets pooled. Each model at its own frozen cut-offs.

* **Primary metric: PR-AUC.**
* Fraud recall, precision, F1 and legitimate alerts per 1,000 at Policy B;
  Critical-tier recall; first-fraud detection; episode detection; ROC-AUC
  (secondary); accuracy and confusion matrix (reported, never used to choose).
* Calibration (Brier score, log loss, expected calibration error, reliability
  table) and score distributions: reported; used only as a tie-break.
* Inference time per transaction and model size: reported; tie-break only.

Intervals follow the multi-seed method already in the project
(`app/evaluation/multiseed.py`): for a difference between two families, the
difference of the means over training seeds 11–15, with a 95% interval that
includes both data variation (group bootstrap within each dataset, 2,000
resamples) and training-seed variation.

## 6. Selection rule

A challenger family is **eligible** only if every condition holds against the
DNN (challenger − DNN):

| # | Condition |
|---|---|
| E1 | PR-AUC: lower bound of the 95% interval **above 0** (it must clearly beat the DNN on the primary metric) |
| E2 | Recall at Policy B: lower bound above −0.05 |
| E3 | Legitimate alerts per 1,000 at Policy B: upper bound below +1.0 |
| E4 | Critical-tier recall: lower bound above −0.05 |
| E5 | Mean recall at Policy B ≥ 0.40 |
| E6 | Stable: PR-AUC higher than the DNN's for at least 4 of the 5 training seeds (same-seed comparison), and higher on at least 4 of the 5 development datasets (mean over training seeds) |
| E7 | Reproducible: refitting with the same seed gives identical scores |

**Winner:** among eligible families, the highest mean PR-AUC. If the two best
eligible families differ by less than 0.005 in mean PR-AUC, the one with the
better calibration (lower Brier score) wins, then the cheaper one to run.

**If no family is eligible, the DNN stays and nothing is replaced.**

Not allowed: choosing on accuracy, on a single seed, on hold-out data, or on
how the scores look.

## 7. The deployed artifact and its confirmation

* Training seed **14**, inherited from Stage B of 4C-3E.6 (chosen there on
  development data, before this protocol). The deployed pipeline is the seed-14
  LSTM, unchanged, followed by the winning family trained with seed 14 on the
  seed-14 out-of-fold Risk Scores, with the setting chosen in §3.
* **Confirmation on the fresh hold-out (501–505), scored once,** against the
  seed-14 DNN (`v2_dnn_lstm_seed14`) and the current production models:
  * C1: PR-AUC difference to the seed-14 DNN, 95% group-bootstrap interval, lower bound above 0;
  * C2: recall at Policy B ≥ 0.40;
  * C3: legitimate alerts per 1,000 difference to the seed-14 DNN, upper bound below +1.0.
* If confirmation fails, the classifier is **not** put into the application.
  The result is reported as it is.

## 8. The application

* If a family is confirmed, it replaces the DNN as the default model set. The
  current production models remain loadable by name.
* The API field `fraud_probability` keeps its name. It is called a calibrated
  probability only if calibration is demonstrated on independent data; this
  protocol expects to keep calling it the **Fraud Score (a model score)**,
  because calibration on synthetic v2 data would not carry over to the
  application's customers or to real data.
* The fixed alert bands (25 / 50 / 80) are not changed.
* SHAP explanations must come from the deployed classifier itself.
