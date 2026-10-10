# Step 4C-2f-2: observability and model consistency

This step makes the platform say which model set scored what, and describe
each model set's scores truthfully.

What it does not change:

* scoring: the numbers and the alert levels are unchanged;
* the production default (`MODEL_SET` unset means production);
* model files, candidate artifacts and datasets (checksums below);
* the `/predict` and `/predict/batch` responses, and the history endpoint.

No candidate was selected, and no threshold was chosen. The threshold and
cold-start questions are analysed only, in
`docs/step4c2f-2-threshold-and-cold-start-analysis.md`.

## 1. Model-set provenance

**Database** (`app/db/models.py`, `app/db/migrations.py`)

* The `transactions` table has two new nullable columns:
  * `model_set`: `production`, `v2_dnn_lstm` or `v2_dnn_only`;
  * `model_version`: a short content-derived ID. For production it is
    `production-<sha12 of the 7 model files' checksums>`; for a candidate it is
    `<name>-<dnn_weights_sha256[:12]>`.
* **Migration.** The project has no migration framework: `create_all` creates
  missing tables but never adds columns. At startup `ensure_schema()` runs
  `ALTER TABLE transactions ADD COLUMN …` for any missing column. This works in
  SQLite and PostgreSQL, and running it again changes nothing.
* **Rows from before provenance** keep `NULL`. They are never back-filled,
  because the model set that scored them is unknown. It could have been a
  candidate between 4C-2e-d and now.
* **Tested on a real old database:** the 32-row database from the 4C-2e-e
  end-to-end run. Both columns were added; the old rows were readable, NULL
  and restored at startup; new rows were recorded.

**Recording**

* `score_transaction` returns `model_set` and `model_version`, and
  `_persist_transaction` stores them.
* Every path records it: `/predict` and batch scoring.
* The `/predict` and `/predict/batch` responses are unchanged: provenance is
  stored, not returned.

**Restart and rollback**

* Restoring from the database replays raw transactions only, as before. The
  stored provenance is never rewritten.
* After production → `v2_dnn_lstm` → `v2_dnn_only` → production on one
  database, each row still carries the model set that scored it. The counts
  were 11 production, 7 `v2_dnn_lstm` and 7 `v2_dnn_only`.

## 2. Model metadata (`app/model_metadata.py`, `GET /model-info`)

This is a new, additive endpoint. It returns the loaded model set's metadata:

* **Identity:** `model_set`, `model_version`, directory, `uses_lstm` and the
  DNN input columns.
* **`model`:**
  * production: the SHA-256 of each file in `models/saved/`, and the legacy
    training recipe;
  * candidates: the manifest SHA-256, the DNN and LSTM weight checksums, the
    seed and the recipe.
* **`dataset`:**
  * production: v1, with the feature file and its SHA-256;
  * candidates: v2, with the dataset and split SHA-256 and the generator
    version.
* **`features`:** the feature list and its hash, plus three version fields:
  * `training_feature_version`;
  * `live_feature_version`, which records the current `build_point_features`
    and the SHA-256 of its source;
  * `training_and_live_features_consistent`.

  The three versions are kept distinct:

  | | Training features | Live features | Consistent |
  |---|---|---|---|
  | production | `legacy-v1` (no std floor, no clipping; see the 4C-2f-1 audit) | current | **false** |
  | candidates | current (v2 generator) | current | true |

* **`score_semantics`:**
  * `fraud_probability` is "DNN output ×100, capped at 99.9 … a fraud SCORE
    from class-weighted training, not a calibrated probability";
  * `risk_score` has a per-model-set meaning;
  * `calibrated_probabilities` is false.
* **`thresholds`:**
  * the 25/50/80 bands, with the status: legacy bands, applied to every model
    set, not derived from any evaluation, candidate thresholds not approved;
  * candidates also carry their manifest thresholds, marked "NOT approved and
    NOT applied by /predict".
* **`cold_start`** and **`evaluation`:** the 4C-2f-1 cold-start policy, and
  what each model set's evaluation actually covers.

## 3. /metrics and /metrics/report

**Production**
* The existing `lstm_risk_predictor` and `dnn_fraud_classifier` entries are
  unchanged.
* New keys are added: `model_set`, `model_version`, `model_metadata`,
  `alerting` (the thresholds block) and `evaluation`.
* `evaluation` records dataset version v1, the report version and dataset,
  `applies_to_loaded_model_set: true`, and "evaluation copies of the
  production architecture retrained on the v1 time split …; not the deployed
  weights in models/saved/".
* `/metrics/report` is the v1 report plus the same model keys.
* A missing report still returns 503.

**Candidates**
* `lstm_risk_predictor` and `dnn_fraud_classifier` are **null**, with
  `v1_metrics_withheld` explaining why. v1 numbers are never shown for a
  candidate.
* `candidate_evaluation` holds that candidate's slice of
  `models/candidates/v2/comparison.json`:
  * the dataset, split, rows and score note;
  * test metrics at the validation-F1 threshold;
  * the FPR operating points, the 25/50/80 band counts and the episode
    summary;
  * the limitations.
* It is given only if `comparison.json` records exactly the loaded weights
  (DNN and LSTM checksums). Otherwise it is `{"available": false, "reason": …}`.
* No new metrics are computed.
* `/metrics/report` has no v1 report: it returns `v1_report_withheld`,
  `candidate_evaluation` and the metadata.

**Compatibility.** Keys were only added. For candidates the two model entries
are null. The frontend already rendered nothing for missing entries, and it
now shows the candidate block instead (§5). One existing test asserted the
exact key set of `/metrics`; it now checks that the two entries are present
and that the model set is labelled.

## 4. PDF (`app/report.py`, `/report/pdf`)

**Where the model set comes from.** `/report/pdf` looks up the transaction's
stored row and uses **its recorded model set**, not the one currently loaded.
So a PDF of an older prediction is still correct after a switch.

**New "Model" section**
* It shows the model set and model version.
* When the row's version is the loaded one, it also shows the training data
  and the model type.

**Wording**

| Model set | Risk Score row | Footer |
|---|---|---|
| production | "Trajectory risk from prior activity, before this transaction: the production LSTM's score for the customer's 10 previous transactions" | production LSTM risk predictor and DNN, with the cold-start note |
| v2_dnn_lstm | "LSTM behavioral-risk component: the v2 LSTM's score for the customer's 10 previous transactions, used as an input to the DNN" | v2_dnn_lstm, synthetic dataset v2, with the cold-start note |
| v2_dnn_only | Labelled **"Risk Score (compatibility)"**: "Same value as Fraud Probability: this model set has no sequence model, so the risk score field repeats the DNN fraud score for compatibility" | DNN on 9 features; "Risk Score repeats the DNN fraud score for compatibility and is not a separate signal". The report contains neither "LSTM" nor "trajectory". |
| not recorded (legacy row, or unknown transaction) | neutral: "its meaning depends on the model set, which was not recorded" | neutral; no model set assumed |

**Other changes**
* In every report, Fraud Probability now reads "This transaction's DNN fraud
  score (0–100); a model score, not a calibrated probability".
* The "Meaning" cells now wrap. Before, the two longest lines overflowed the
  table.

## 5. Frontend wording

The frontend fetches `GET /model-info` once. The wording lives in
`src/modelWording.js`. If `/model-info` fails, the wording is neutral.

| Place | production | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| Live Scan: new line above the gauges | "Loaded model set production (production-…)" | same pattern | same pattern |
| Risk gauge label | Risk Score | Risk Score | **Risk Score (= fraud score)** |
| Risk gauge caption | "Trajectory risk from this customer's prior activity (LSTM), before this transaction" | "LSTM behavioral-risk component from this customer's previous transactions, …" | "This model set has no sequence model: Risk Score repeats the DNN fraud score for compatibility" |
| Fraud Probability caption | "…fraud score (model output, not a calibrated probability) -- can be high even if prior trajectory was clean" | same, "prior behavior" | "This transaction's DNN fraud score (model output, not a calibrated probability)" |
| Timeline risk line | Risk Score (LSTM) | Risk Score (LSTM) | Risk Score (= fraud score) |
| Batch table column | Risk Score | Risk Score | Risk Score (= fraud score) |
| Model Performance tab | labelled "Loaded model set production … · v1 evaluation"; the note now says evaluation copies, not calibrated | the v1 blocks are replaced by a note that the v1 evaluation does not apply, and the candidate's v2 test-period block from `/metrics` `candidate_evaluation` (threshold not applied by live scoring) | same as v2_dnn_lstm |

**Unchanged**
* The layout and components (no redesign).
* The API calls, apart from the new `GET /model-info`.
* The `RiskGauge` colours, which use the 25/50/80 stops. They are cosmetic;
  threshold policy is still undecided.

**Known limitation.** The timeline plots stored rows from every model set under
the current model set's line name. The history endpoint was deliberately left
unchanged. Showing per-point provenance would need an additive field there,
which is a decision for later.

**Verification**
* The project has no frontend test framework, so none was added.
* `npm run lint` (oxlint) is clean.
* `vite build` succeeds. It was built to a scratch directory, so the repo's
  `frontend/dist` was not touched.
* The UI was verified in the browser for all three model sets (§7).

## 6. Other wording

* **OpenAPI:** the `PredictionResponse.risk_score` and `fraud_probability`
  descriptions are now model-set-aware and state "not a calibrated
  probability".
* **Database column comments** were updated to match.
* **The `/metrics` docstrings** describe the new behaviour.

## 7. Verification

**Backend tests**
* `tests/test_observability.py` (27 tests):
  * migration on an old-schema, a new and an empty database;
  * provenance for `/predict`, batch, both candidates and restore (no
    back-fill);
  * `/model-info` for all three model sets;
  * `/metrics` and `/metrics/report` labelling, and that candidates never get
    v1 numbers;
  * the `comparison.json` slice, and refusal when the weights differ;
  * PDF wording for all three model sets, both with stand-in and with real
    candidates;
  * a PDF naming the recorded model set, not the loaded one;
  * neutral PDFs for legacy and unknown rows.
* `tests/test_cold_start_sensitivity.py` (28 tests): pins the current
  unusual-hour and unusual-category rule (analysis only).
* `tests/test_metrics.py`: one test updated, as described in §3.

**End-to-end** (live uvicorn + Vite + Playwright)

1. The legacy database was migrated, with rows NULL and a neutral PDF.
2. Production, `v2_dnn_lstm` and `v2_dnn_only` were each run on one shared
   database, with restarts between them. For each:
   * `/model-info`, `/metrics` and `/metrics/report` were labelled;
   * `/predict` and batch scoring worked, with 14 response fields;
   * the PDF wording was correct, including PDFs of rows scored earlier by
     other model sets;
   * the database recorded the model set on every new row;
   * the UI showed the loaded model set, the model-aware captions, the batch
     header and the Model Performance tab, with no page errors.
3. Rollback to production restored all 21 rows with their provenance intact.
4. Invalid `MODEL_SET` values still stop startup.

Full suite, `git diff --check` and checksums: see the step report.
