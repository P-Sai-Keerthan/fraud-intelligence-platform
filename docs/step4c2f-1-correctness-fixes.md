# Step 4C-2f-1: production correctness fixes

This step fixes three correctness problems that the 4C-2e-e readiness audit
found. They affect every model set, including production:

* back-dated transactions were scored with the wrong row;
* the cold-start path fed models inputs they had never seen in training;
* the home device ignored the approved 90-day rule.

The v1 feature-version question is audited separately in
`docs/step4c2f-1-v1-feature-version-audit.md`.

What this step does not change:

* the production default (`MODEL_SET` unset means production);
* any model file (`backend/models/saved/` is byte-identical before and after);
* the API schema, database schema, PDF, `/metrics`, alert thresholds and
  frontend wording.

No candidate was selected.

## 1. Back-dated transactions

### Root cause

`score_transaction` did the following:

1. It added the new row to the end of the customer's raw history.
2. It rebuilt the features with `build_point_features`, which sorts the
   history by timestamp.
3. It scored `recomputed.iloc[-1]`, assuming the new row was still last.

When the request's timestamp was earlier than the customer's newest
transaction, the sort moved the new row and the pipeline scored another
transaction. The similarity score came from that wrong row too.

There was a second defect. `is_new_device` and `is_new_location` for the new
row were computed against the customer's whole history, including
transactions dated after it. Transactions after an inserted row also kept
flags that were now stale.

### Fix (`app/inference_pipeline.py`)

**Finding the right row**
* The transaction's own row is located by its `transaction_id`. If that ID is
  not found exactly once, the pipeline raises `RuntimeError`, so it never
  returns another transaction's score.
* The transaction's features, its LSTM window (the 10 transactions before it)
  and its similarity all use only the transactions strictly before it in
  time.

**Inserting at the right place: `insert_chronologically`**
* This function puts new rows at their place in time with a stable sort:
  * existing rows keep their order;
  * a new row goes after existing rows that have the same timestamp.
* It recomputes `is_new_device` and `is_new_location` as "not used by any
  earlier transaction" for every row from the inserted position onward.
  Earlier rows cannot be affected by the insertion.
* In both the v1 seed data (93,913 rows) and the v2 data (101,297 rows), the
  stored flags equal exactly this definition. So for a normal, newest
  transaction nothing changes.

**Restart consistency**
* `restore_scored_history` (run at startup from the database) inserts rows
  through the same function, including rows dated before seed rows.
* A restart therefore rebuilds exactly the history the live service had. The
  tests compare the two row by row, including after several back-dated
  insertions.

**Transaction IDs**
* A generated ID that already exists in the customer's history is
  regenerated.
* After 5 collisions the pipeline raises `RuntimeError`, and nothing is
  added to the history.
* Rows restored from the database with a duplicate ID are skipped (unchanged
  behaviour).

**Normal newest-transaction scoring is unchanged.** 60 scorings of seeded
customers produced byte-identical responses before and after the change,
including the SHAP reasons (seeded).

### Limitations

**Exact timestamp ties**
* When a transaction has exactly the same timestamp as an existing one, the
  ordering is the feature builder's (a non-stable sort by timestamp). The
  score returned is still the one computed for the requested row.
* v1 has 426 rows with within-customer timestamp ties; v2 has none.

**Timezones**
* Mixing timezone-aware and naive timestamps for one customer fails, as it
  did before this step.

**ID uniqueness scope**
* IDs are checked against the customer's in-memory history. Global
  uniqueness is still enforced by the database primary key.

### Regression tests (`tests/test_backdated.py`, 10 tests)

**Reference used by the tests.** Every scoring is compared with a reference:
the same transaction scored as the newest transaction of a history that holds
exactly the customer's earlier transactions.

| # | Case | Before the fix | After |
|---|---|---|---|
| 1 | Normal newest transaction | pass | pass |
| 2 | Transaction inserted between two existing ones (it lands between them; history matches a replay) | fail | pass |
| 3 | Transaction earlier than the customer's latest: the ₹85,000 Lagos case, directly and through `/predict` (Critical, persisted with its own score) | fail ×2 | pass |
| 4 | Duplicate transaction ID: regenerated; fails clearly when no unique ID is possible; replayed duplicates skipped | fail ×2, pass ×1 | pass |
| 5 | Multiple back-dated transactions in shuffled order; a back-dated new device updates the later row's flag | fail ×2 | pass |
| 6 | The returned score and echoed fields belong to the requested transaction (5 transactions in scrambled time order) | fail | pass |

**Before the fix:** 8 failed, 2 passed. **After:** 10 passed.

## 2. Cold start

### Trace

**History needed: 10 earlier transactions (`SEQUENCE_LENGTH`), for every model**

* The LSTMs were trained on windows of 10 real earlier rows:
  * production: `build_sequences`, which uses targets from index 10 of
    customers with more than 10 rows;
  * candidates: `evaluation.windows.build_windows`, with the same rule.
* The DNNs were trained only on those window targets:
  * production's `dnn_model.py` inner-joins the features with the LSTM
    metadata. Its comment mentions a median fill, but the code drops the
    first 10 rows of each customer.
  * the candidates use the evaluation windows.
* So no model has ever seen a transaction with fewer than 10 earlier
  transactions.

**Padding (before this step)**
* With 0 earlier transactions, the LSTM window was all zeros.
* With 1–9, the earliest row was repeated at the front.
* This matched `feature_engineering.get_latest_sequence_for_customer`. That
  function is not used by the pipeline, and it pads the same way.

**How the scalers interact with padding**
* Windows are scaled with `(x − mean) / std`, so a zero row does not stay
  zero. For example, `amount_pct_of_avg` becomes −2.32 for production and
  −1.04 for A.
* After scaling, the full zero row is:
  * production: [−0.013, −0.149, −0.102, −0.056, −0.073, −0.247, −0.125, −0.389, −2.322];
  * A: [−0.042, −0.532, −0.081, −0.066, −0.067, −0.160, −0.432, −0.327, −1.043].

**The Masking layer is not effective**
* Both LSTMs (production and A) have `Masking(mask_value=0.0)` as their
  second layer. It masks a timestep only if every scaled feature is exactly
  0.
* That never happens, as shown above for zero rows. Repeated rows are real
  rows, which are not zero either.
* So padded steps are fed to the LSTM as if they were real transactions.

**Features for a customer's first transaction.** `build_point_features` uses
placeholders when there is no history:

| Feature | Placeholder value |
|---|---|
| `is_new_device` | 1 (no earlier rows) |
| `is_new_location` | 1 |
| `category_is_unusual` | 1 |
| `amount_zscore` | 0 (compared with itself) |
| `amount_pct_of_avg` | 100 |
| `hour_is_unusual` | 0 |

After scaling:
* both new-device and new-location flags are at the +6 clip for every model
  set;
* `category_is_unusual` is +6 for production and +2.44 for the candidates.

**Training versus inference**

| | Training | Inference before this step |
|---|---|---|
| Earlier transactions | always 10 or more | 0 or more |
| LSTM window | 10 real rows | zeros or repeated rows |
| First-transaction placeholders | never seen | used as real values |

**Which features drove the early false alerts.** Each input was replaced by
its training mean (scaled 0) in turn, on the 4C-2f-1 trace customer (a
consistent ₹2,000 grocery buyer). The table shows the largest drops in the
score, in percentage points.

| Model set | 0 earlier transactions | 1–2 earlier transactions |
|---|---|---|
| production (before: 36.98, Medium) | `category_is_unusual` −34.1, `is_new_device` −28.5 | `risk_score` is 99.62 / 97.11 from the padded windows. In the 4C-2e-e scenario this produced Critical (97.2) at 1 earlier transaction. |
| A (before: 75.6, 60.4 and 60.6, all High) | `is_new_device` −31.0, `is_new_location` −11.6 | `risk_score` −52.4 / −51.6 (`risk_score` 97.7 / 94.4 from padding) |
| B (before: 67.6, High) | `is_new_device` −38.2, `is_new_location` −3.6 | none: already Low |

So the drivers were:
* the padded LSTM windows, at 1–9 earlier transactions;
* the first-transaction placeholders, at 0 earlier transactions.

### Fix

This is the smallest change that stops presenting unsupported inputs as
normal, fully contextual scoring. No new model was added and nothing was
retrained. It uses what the architecture already provides: every DNN input is
standardized, so a scaled value of 0 is that input's training mean. That is
the standard neutral imputation for a missing input.

1. **Fewer than 10 earlier transactions (`MIN_PRIOR_TRANSACTIONS = SEQUENCE_LENGTH`)**
   * The LSTM is not run. No window is fabricated.
   * The DNN's `risk_score` input is set to scaled 0, which is the training
     mean.
   * The response's `risk_score` reports that mean, rounded:

     | Model set | Reported `risk_score` |
     |---|---|
     | production | 13.6 |
     | A | 29.85 |
     | B | unchanged, `risk_score` = `fraud_probability` |

2. **No earlier transaction at all**
   * The six baseline-relative features are treated as missing and set to
     scaled 0: `amount_zscore`, `hour_is_unusual`, `is_new_device`,
     `is_new_location`, `category_is_unusual`, `amount_pct_of_avg`.
   * The baseline-free features are used as observed: `is_foreign_location`,
     `failed_logins_24h`, `txn_velocity_1h`.
   * So a first transaction from a foreign city with failed logins still
     scores higher than an ordinary one, for every model set (tested).
     Nothing is hard-coded to Low.
3. **Imputed inputs are never reported as SHAP reasons**
   (`FraudExplainer.explain(..., exclude=...)`).
4. **The pipeline records the state** in `result["history_context"]`: the
   number of earlier transactions, whether it is a cold start, whether the
   LSTM was used, and which inputs were imputed. This is internal and not
   part of the API response, so the schema is unchanged. Surfacing it (for
   example with provenance) is for 4C-2f-2 or later.

With 10 or more earlier transactions nothing changes. Every seeded customer
has at least 62.

**Before and after, 4C-2e-e scenario** (next normal transaction after k
earlier transactions; score and alert level):

| k | production before | production after | A before | A after | B before | B after |
|---|---|---|---|---|---|---|
| 0 | 37.0 Medium | 1.9 Low | 75.6 High | 13.8 Low | 67.6 High | 17.7 Low |
| 1 | 97.2 Critical | 36.1 Medium | 78.1 High | 11.0 Low | 16.1 Low | 16.1 Low |
| 2 | 3.7 Low | 1.4 Low | 60.6 High | 9.0 Low | 10.2 Low | 10.2 Low |
| 5 | 2.2 | 1.4 | 9.2 | 9.1 | 10.3 | 10.3 |
| 9 | 2.8 | 1.2 | 13.4 | 10.3 | 12.0 | 12.0 |
| 10 and more | unchanged | unchanged | unchanged | unchanged | unchanged | unchanged |

### Residual (not changed; decision for the project owner)

With 1–9 earlier transactions, the point features are real measurements, so
they are kept. They are still statistically weak:

* `hour_is_unusual` and `category_is_unusual` fire whenever the value's share
  of the customer's history is below 5%. With fewer than 20 earlier
  transactions, any hour or category not seen before counts as unusual.
* The remaining production result at k = 1 (36.1 Medium) comes from
  `hour_is_unusual` = 1, because 12:30 differs from the single earlier 13:15.
  Removing it lowers the score by 32 percentage points.

Options for later:
* also impute the baseline-relative features when there are fewer than 10
  earlier transactions;
* add a minimum-history policy;
* retrain with early-history rows.

These rows are marked `cold_start` in `history_context`.

### Tests (`tests/test_cold_start.py`, 27 tests)

For every model set (production, A and B, using the real local candidates),
with k = 0, 1, 2, 5, 9, 10 and 15 earlier transactions, the tests check:

* the history holds exactly k + 1 rows (nothing fabricated);
* the `history_context` values;
* the LSTM is called 0 times below 10 earlier transactions and once from 10;
* the exact scaled DNN inputs: imputed inputs are 0, and with 10 or more
  earlier transactions the inputs are the observed ones;
* the reported `risk_score`;
* imputed inputs never appear as reasons.

Further tests check:
* that a suspicious first transaction scores higher than a normal one, and
  that its reasons come only from baseline-free features;
* that seeded customers are never in cold start;
* that the API response has the same 14 fields.

## 3. 90-day home device

**Finding.** `get_customer_profile` used the most frequent device over the
whole history. It did not follow the approved rule in
`docs/step4c2e-retraining-design.md` §5.

**Implemented, as designed**
* `home_device` is the most frequent device in the window (anchor − 90 days,
  anchor]. The anchor is the customer's newest transaction, not the wall
  clock. Ties are broken alphabetically.
* The rule falls back to the whole history when the window has fewer than 5
  transactions, or when the history spans less than 90 days.
* A customer with no history still gets `None`, so `/profile` returns 404
  and batch scoring reports a row error.
* `home_location` keeps the whole-history rule. The design marked it "to be
  confirmed", so it is not changed here and is listed as a decision.
* The `CustomerProfile.home_device` description in the OpenAPI schema was
  updated. The home device is not a model input. It affects only
  `GET /customer/{id}/profile` (the form defaults) and the device batch
  scoring fills in when a row leaves it blank.

**Effect**
* v1 (the live data): identical for all 500 customers. This is tested, and
  the existing profile tests pass unchanged.
* v2: the 32 phone upgraders would get their new phone. v2 is not live.

**Tests (`tests/test_home_device.py`, 17 tests)**
* more than 90 days on a stable device;
* an upgrade inside the window, both once the new phone dominates and before
  it does;
* an upgrade before the window;
* an old burst of use outside the window, which is ignored;
* less than 90 days of history, which uses the whole history;
* spans of 89, 90 and 91 days;
* the window boundary is half-open;
* a sparse window, which falls back to the whole history;
* alphabetical tie-breaking;
* no history;
* location still uses the whole history;
* the batch default uses the 90-day device;
* all 500 seed customers are unchanged.

## 4. Verification

* **Tests:** the new tests pass (10 + 27 + 17 = 54). The full backend suite
  passes 371 of 371 (317 before this step), with no skips.
* **Production pipeline:** unchanged for seeded (newest) transactions. 60 of
  60 responses were byte-identical to the original code; the only difference
  was the intended cold-start change for a brand-new customer.
* **Production model SHA-256** (`backend/models/saved/`): identical before and
  after. See the final report.
