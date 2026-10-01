# Step 4C-2f-2: threshold and cold-start sensitivity analysis

This is analysis only. No threshold, band or feature behaviour was changed.

## Caveats that apply to every number below

* **The scores are not calibrated probabilities.** Every DNN was trained with
  class weights, so "85% fraud probability" is a model score, not an 85%
  chance.
* **25/50/80 are legacy alert bands.** `/predict` applies them to every model
  set. They were not derived from any evaluation of the loaded model.
* **Candidate-specific thresholds have not been approved.** They are recorded
  in the manifests, chosen on the v2 validation split only, and are not
  applied by `/predict`.
* **The threshold policy remains an owner decision** (4C-2e-e D4).

**What "live v1" means:** all 88,913 full 10-transaction windows of the live
v1 customer history, with features computed by the current code, exactly as
live inference sees them. The v1 fraud labels are trivially separable, and
production was trained on this data, so live-v1 numbers show **alert volume**,
not detection quality.

**Validation FPR** is on the v2 validation split. Some rows are slightly above
target because the FPR helper admits tied scores (a known issue, 4C-2e-c).

## Production: legacy 25/50/80 bands

| Band | Live v1, per 1,000 windows |
|---|---|
| Medium (25–50) | 3.36 |
| High (50–80) | 1.18 |
| Critical (≥ 80) | 9.58 |
| High + Critical | 10.76, of which 1.55 legitimate (in-sample) |

Production has no manifest and no validation thresholds of its own. The v1
evaluation's thresholds (DNN 0.542, LSTM 0.998) belong to evaluation copies
and are not applied by `/predict`.

## v2_dnn_lstm (candidate A)

| Operating point | Threshold (score ×100) | v2 validation FPR | v2 test: alerts per 1,000 / precision / recall | Live v1: alerts per 1,000 (of which legitimate) |
|---|---|---|---|---|
| Validation F1 | 0.9593 (95.93) | 0.0029 | 2.32 / 0.184 / 0.148 | 5.15 (0.00) |
| FPR 0.1% | 0.9679 (96.79) | 0.00105 | 0.95 / 0.200 / 0.066 | 5.03 (0.00) |
| FPR 1% | 0.7677 (76.77) | 0.0100 | 8.04 / 0.147 / 0.410 | 9.16 (0.11) |
| FPR 5% | 0.5894 (58.94) | 0.0500 | 48.53 / 0.038 / 0.639 | 20.67 (11.46) |
| Legacy bands, High + Critical | 50 | — | 65.27 | 32.55 (23.34) |

## v2_dnn_only (candidate B)

| Operating point | Threshold (score ×100) | v2 validation FPR | v2 test: alerts per 1,000 / precision / recall | Live v1: alerts per 1,000 (of which legitimate) |
|---|---|---|---|---|
| Validation F1 | 0.7698 (76.98) | 0.0106 | 9.46 / 0.120 / 0.393 | 8.98 (0.06) |
| FPR 0.1% | 0.9119 (91.19) | 0.00105 | 1.18 / 0.400 / 0.164 | 6.59 (0.00) |
| FPR 1% | 0.7744 (77.44) | 0.0100 | 9.27 / 0.122 / 0.393 | 8.95 (0.04) |
| FPR 5% | 0.5765 (57.65) | 0.0500 | 52.27 / 0.035 / 0.639 | 15.08 (5.87) |
| Legacy bands, High + Critical | 50 | — | 81.26 | 27.97 (18.76) |

## Observations (no recommendation)

**The bands and the operating points do not line up**
* Under the legacy bands, both candidates put tens of transactions per 1,000
  into High + Critical, on both datasets.
* Every validation-chosen operating point except FPR 5% sits inside High or
  Critical. A's F1 point (95.93) is deep inside Critical.
* So "Critical" means something different for each model set. A band policy
  and an alert policy may need to be separate.

**The same threshold behaves differently on v1 and v2**
* On live v1, the candidates' tight operating points (F1, FPR 0.1%) flag
  almost no legitimate transactions. Their alert counts come almost entirely
  from v1's easy fraud rows.
* The FPR-5% points produce 6–11 legitimate alerts per 1,000.
* The same threshold therefore gives very different alert volumes on v1 and
  v2, because the score distributions shift between datasets (4C-2e-e §3.2).

**The display suggests a probability that isn't there.** Fraud Probability is
shown as a percentage on uncalibrated scores. Any threshold policy should
decide whether to calibrate the scores or relabel the display.

## Cold-start sensitivity: unusual hour and unusual category

**The current rule** (`build_point_features`; unchanged)

A value (the hour of day, or the merchant category) is "unusual" when its
share of the customer's earlier transactions is below 5%. With *n* earlier
transactions:

| n earlier | `hour_is_unusual` | `category_is_unusual` | Pinned by `tests/test_cold_start_sensitivity.py` |
|---|---|---|---|
| 0 | 0 (no hours seen) | 1 (placeholder) | yes. Both are imputed as missing by the pipeline since 4C-2f-1. |
| 1 | any hour not seen before → 1 | any category not seen before → 1 | yes |
| 2 | same | same | yes |
| 3–9 | same: a value seen once is normal, an unseen one is always unusual | same | yes (3, 5, 9) |
| 10–20 | same (1/n ≥ 5% while n ≤ 20) | same | yes (10, 15, 20) |
| 21 and more | a value seen only once is now also unusual (1/21 < 5%); one seen twice is normal up to n = 40 | same | yes (21, 30, 40) |

**Consequence.** With fewer than about 20 earlier transactions, any hour or
category the customer has not used before is flagged, however varied their
behaviour.

**Measured on legitimate transactions** (current feature code; percentage of
rows with the flag set):

| n earlier | Rows (v1 live) | v1 unusual hour | v1 unusual category | v2 unusual hour | v2 unusual category |
|---|---|---|---|---|---|
| 0 | 500 | 0.0% | 100% | 0.0% | 100% |
| 1 | 500 | 76.8% | 74.0% | 87.2% | 82.4% |
| 2 | 500 | 60.2% | 55.2% | 77.8% | 62.8% |
| 3–9 | 3,500 | 24.4% | 20.1% | 53.9% | 39.1% |
| 10–19 | 5,000 | 4.1% | 2.5% | 25.9% | 16.8% |
| 20 and more | 83,094 | 0.2% | 0.1% | 17.1% | 13.5% |

* **Every model was trained only on rows with 10 or more earlier
  transactions** (4C-2f-1). It never saw the 60–87% flag rates of the first
  few transactions.
* **The example from 4C-2f-1 remains.** A production customer with one
  earlier transaction still scores 36% (Medium), because of
  `hour_is_unusual` = 1 (12:30 versus the single earlier 13:15).
* **These flags are left as they are.** They are real measurements, not
  placeholders. Neutralising them below 10 or 20 earlier transactions, adding
  a minimum-history policy, or retraining with early-history rows is an owner
  decision.
