# Step 4C-2f-1: v1 feature-version audit

This is an audit only. Nothing was regenerated or retrained, and production
behaviour is unchanged.

## Question

4C-2e-e found that recomputing the committed v1 feature file
(`data/transactions_with_features.csv`) with the current
`build_point_features` changes `amount_zscore` on 43,442 of 93,913 rows, and
`amount_pct_of_avg` on 4 rows.

## Old and current behaviour

| | Old (produced the committed CSV) | Current (`feature_engineering.py`, identical since the initial commit afbbce8) |
|---|---|---|
| Standard deviation of earlier amounts | `std` over at least 2 earlier amounts; with 0 or 1 earlier amounts, `max(0.3 × avg, 1)` | the same, then floored: `max(std, 0.1 × avg, 1.0)` |
| `amount_zscore` | not clipped (the CSV has values up to 1,163) | clipped to ±10 |
| `amount_pct_of_avg` | not clipped (up to 8,440) | clipped to [0, 1000] |
| Every other feature | identical | identical |

## Evidence

**Which implementation produced the CSV**
* Re-implementing the old rule reproduces the committed CSV:
  * `amount_pct_of_avg` matches on every row;
  * `amount_zscore` matches on every row but 3, which differ by at most
    1.6 × 10⁻⁵ (floating-point summation order).
* The current rule differs on 43,441 rows (43,442 by the 4C-2e-e
  full-pipeline recompute, one of which is a summation-order tie).
* The current code's own comment explains the change: "customers with
  unusually consistent spending … z-scores (seen up to ±1000+ in practice)".
* The CSV and `feature_engineering.py` were both added in the initial commit
  (afbbce8). The CSV was generated before the floor and clips were added, by
  an uncommitted earlier version of the code.

**Causes**
* The floor binds on 43,388 rows: 46% of rows, mostly customers whose
  spending barely varies.
* 149 rows have |z| > 10 in the old file.
* 4 rows have `amount_pct_of_avg` > 1,000.
* The rows that differ are almost all legitimate: 43,362 legitimate and 79
  fraud.
* Median absolute change in z: 0.21. 90th percentile: 0.86. 99th percentile:
  2.17.

**What production training used: the old values**
* Rebuilding the production training inputs from the committed CSV with the
  production training code reproduces the saved scalers exactly:
  * `train_test_split(random_state=42, stratify=y)` gives
    `lstm_feature_mean.npy` and `lstm_feature_std.npy` with a maximum
    difference of 0.0;
  * the first 9 entries of `dnn_feature_mean.npy` and `dnn_feature_std.npy`
    also have a maximum difference of 0.0.
* With current-code features they do not match. For example, the DNN's
  standard deviation of `amount_zscore` is 2.03 as saved, but 1.02 from
  current-code features.

**What live inference uses: the current code**
* `score_transaction` recomputes every row of the customer's history from the
  raw columns with the current `build_point_features`.
* The stale feature columns loaded from the CSV into the seed history are
  never used for scoring.

## Consequences

**Production (live)**
* Production has a training/serving difference on `amount_zscore` (and on
  `amount_pct_of_avg` for extreme rows): the scalers and weights were fitted
  on unfloored, unclipped values, while live inputs are floored and clipped.
* Measured on all 88,913 v1 windows, comparing the old CSV features with the
  current-code features, both scored by production:

  | Measure | Result |
  |---|---|
  | `fraud_probability`, largest change | 21 percentage points |
  | `fraud_probability`, 99th percentile of changes | 2.0 percentage points |
  | `risk_score`, largest change | 11.2 |
  | Alert level changed | 19 rows, all legitimate |

  The 19 alert changes are: 11 Low → Medium, 7 High → Medium, 1 Medium → Low.
* No fraud row changed alert level.

**`/metrics` (v1 evaluation, 4B)**
* It evaluates evaluation copies trained and tested on the old CSV values,
  not on the values live inference computes.

**Candidates and the 4C-2d / 4C-2e evaluation: not affected**
* The v2 generator builds its features with the current code. The sanity
  audit recomputed all 101,297 rows with a maximum difference of 0.
* The candidates and their evaluation used only v2.

**The v2 generator** is not affected.

**Tests that depend on the old values**
* `test_dataset_versions.py` asserts the SHA-256 of the committed CSV.
* `test_evaluation.py` and `test_profile.py` read the CSV. They use its IDs,
  labels and devices, not the amount features.
* The saved v1 evaluation report (the 4C-2c v1 regression and `/metrics`) can
  be reproduced only from the old values.
* So regenerating the CSV would break the hash test and invalidate the saved
  v1 evaluation report. Live-scoring tests do not depend on the old values,
  because live scoring always recomputes features.

## Recommendation (not done; decision for the project owner)

1. **Keep the committed v1 CSV as it is** for now, as the record of what
   production and the v1 evaluation were trained on. Do not regenerate it
   silently.
2. **If production is ever retrained on v1**, regenerate the v1 features with
   the current code first. Then re-run the v1 evaluation and update the hash
   test, so that training, evaluation and live inference use one feature
   version.
3. **Record a feature version** alongside every model and evaluation, for
   example a SHA-256 of the feature code, or a version constant checked at
   load time. This would catch this kind of drift automatically. Candidate
   manifests already pin the feature list; they do not pin the feature code.
4. **Treat the difference as a known production limitation** in the
   production-switch decision (4C-2e-e, D1/D8). It is small in alert terms
   (19 of 88,913 rows), but it is a real training/serving mismatch.
