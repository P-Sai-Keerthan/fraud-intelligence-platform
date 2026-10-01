# Step 4C-2e-e: production-switch readiness audit

This audit checks whether the application is ready to switch production to either
v2 candidate. It changes nothing:

* `MODEL_SET` still defaults to `production`;
* no model set was switched and no winner was chosen;
* no application code, model file or frontend file changed;
* `backend/models/saved/` is byte-identical before and after (§12).

Everything below was measured on the local artifacts from 4C-2e-b to 4C-2e-d
(scripts and outputs are in the session scratchpad, not the repository). The
candidates are:

* **A** = `v2_dnn_lstm` (LSTM risk score feeding the DNN);
* **B** = `v2_dnn_only` (DNN on the 9 features, no LSTM).

## 1. Readiness matrix

| Area | Status | Summary |
|---|---|---|
| MODEL_SET | **READY** | Production is the default. Candidates must be chosen explicitly. Invalid values stop startup, and there is no fallback (§2). The candidate artifacts are untracked local files, so deploying a candidate also means shipping them (decision D7). |
| Production safety | **READY** | `models/saved/` is never written. Its checksums are unchanged. Production responses are byte-identical to the pre-4C-2e-d code (§2). |
| Real-history compatibility | **NOT READY** | The feature names, order and code match. However: (a) a pre-existing bug scores a back-dated transaction with the wrong row, in every model set; (b) cold-start transactions raise false High alerts for A and B, and for production; (c) on the live v1 history, the candidates raise 19–23 legitimate High+Critical alerts per 1,000 under the current bands (§3). |
| risk_score semantics | **NOT READY** for B. **READY** for production and A. | For B, `risk_score` equals `fraud_probability`, but the UI and PDF describe it as trajectory (LSTM) risk (§4). |
| PDF | **NOT READY** for B. **Needs wording** for A. **READY** for production. | The PDF has no knowledge of the model set, so it always uses the production wording (§5). |
| /metrics | **NOT READY** for A and B. **READY** (unchanged) for production. | `/metrics` always serves the v1 evaluation report. It does not identify the model set, model version, dataset or alert configuration (§6). |
| Thresholds | **NEEDS DECISION** | The fixed 25/50/80 bands give 65 (A) and 81 (B) High+Critical alerts per 1,000 on the v2 test period, and 33 / 28 per 1,000 on the live v1 history (§7). |
| SHAP | **READY** | Feature names follow the model set and are validated. B never names `risk_score` as a reason. Explanations are still stochastic, as before (§9). |
| Batch | **READY** | Same fields, validation and summary for all model sets (§9). |
| Database | **READY** for persistence. **NEEDS DECISION** for provenance. | Rows persist and restore for every model set, but no row records which model set scored it (§9). |
| Frontend | **READY** for the response shape. **NOT READY** for B's wording. | It renders every model set without errors, but the captions describe B's `risk_score` as trajectory risk, and the Model Performance tab shows v1 metrics (§9). |
| Rollback | **READY** | Restart with `MODEL_SET` unset or `production` (about 5 s). Scores then match production exactly. Stored history from a candidate period stays unlabelled (§9). |
| Performance | **READY** | Load time, memory and latency are within about 25% of production for both candidates (§10). |
| Evaluation evidence | **NEEDS DECISION** | The evidence is too thin to decide a switch on its own (§8). |

## 2. MODEL_SET safety

**Default and explicit selection**
* With `MODEL_SET` unset, `models/saved/` is loaded. This was checked by the
  test `test_default_is_production` and by live server restarts.
* A candidate loads only when named exactly: `v2_dnn_lstm` or `v2_dnn_only`.

**Invalid values**
* These invalid values were checked with a live server: `v2_DNN_ONLY` and the
  empty string. In both cases uvicorn exits with code 3 and prints:
  `ModelSetError: invalid environment variable MODEL_SET value …; expected one of [...]`.
  The tests cover 7 invalid values.

**No fallback**
* A requested candidate with a missing directory or manifest stops startup.
  This is checked by `test_requested_candidate_never_falls_back_to_production`.
* The same applies to a missing file, a checksum or weights mismatch, a feature
  or order mismatch and a shape mismatch. 29 further tests in
  `test_model_sets.py` cover these cases.

**Production artifacts untouched**
* The loader only reads files, and a candidate root inside `models/saved/` is
  refused.
* The SHA-256 of all 7 production files matched before and after this audit
  (§12). This was checked in the cloud copy and on the user's machine.

**Rollback**
* Rollback is a restart without `MODEL_SET`, or with `MODEL_SET=production`.
  Startup took 4.6–5.2 s for every model set.
* After rolling back on a fresh database, production gave byte-identical
  `/predict` and batch results to the production run before any switch:
  * normal: 23.89 / 1.78 / Low;
  * batch rows: 33.47 / 99.9 / 8.29.

**Deployment gap**
* `models/candidates/v2/` (732 KB) and `models/evaluation/v2/` (8.5 MB) are
  untracked, local-only files. They are not in `.gitignore`.
* A fresh clone with `MODEL_SET=v2_*` fails at startup. This is safe, but it
  means the artifacts have to be shipped deliberately (decision D7).

## 3. Real production history compatibility

The live application scores against the v1 seed history:
`data/transactions_with_features.csv`, with 500 customers and at least 62
transactions per customer. It also replays scored transactions stored in the
database. The live dataset is not switched to v2.

### 3.1 Features

**Names and order match**
* The 9 feature names and their order are identical for every model set. The
  loader checks this against each manifest.

**The live features match the candidates' training features**
* `score_transaction` recomputes the whole customer history from the raw
  columns with the current `build_point_features` code. The v2 training data
  was built with that same code.

**Finding: the v1 seed CSV was built with older feature code**
* Recomputing all 93,913 v1 rows with the current code changes
  `amount_zscore` on 43,442 rows (maximum difference 1,583). It also changes
  `amount_pct_of_avg` on 4 rows (maximum difference 7,440). The current code
  clips these to ±10 and [0, 1000]; the CSV has values up to 1,163 and 8,440.
* The production models and the v1 evaluation (`/metrics`) were trained and
  evaluated on the old values.
* Live inference, for every model set, uses the current values.
* This is an existing training/serving difference for production and does not
  affect the candidates. Rescoring the live history with the recomputed
  features changes production's High+Critical rate only slightly (10.84 → 10.76
  per 1,000). This was not fixed.

**Range**
* Scaled with each candidate's scaler, 1.0% of live v1 rows have any input
  beyond ±6, the level at which inputs are clipped.
* Production on v2 test rows has 26.7% beyond ±6, mostly `hour_is_unusual` and
  `category_is_unusual`. Those two flags are far rarer in v1 (2.9% and 2.7% of
  rows) than in v2 (19.5% and 15.6%).
* So the candidates are not pushed outside their training range by v1 history.

### 3.2 Scores on the live (v1) history

All 88,913 full 10-transaction windows were scored, using the current-code
features. The v1 fraud labels are trivially separable, and production was
trained on this data (in-sample), so these rows are an alert-volume check
only, not a performance comparison.

| Model set | Median legitimate score | Legitimate High+Critical per 1,000 | Alerts per 1,000 at the manifest F1 threshold |
|---|---|---|---|
| production | 0.019 | 1.55 (in-sample) | — (no manifest) |
| A | 0.115 | 23.34 | 5.15 (0 legitimate) |
| B | 0.108 | 18.76 | 8.98 (0.06 legitimate) |

The candidates' scores are shifted upwards on the live customers. Under the
current bands, a demo or production user on the v1 history would see roughly
12–15 times more High alerts than with production.

### 3.3 Cold start and missing history

A new customer was scored through the real pipeline. Each earlier transaction
was a consistent grocery purchase of about ₹2,000 on the same device and city.
The table shows the next normal transaction, as `fraud_probability` and alert
level.

| Earlier transactions | 0 | 1 | 2 | 5 | 9 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| production | 37.0 Medium | **97.2 Critical** | 3.7 Low | 2.2 Low | 2.8 Low | 3.4 Low | 2.0 Low |
| A | **75.6 High** | **78.1 High** | **60.6 High** | 9.2 Low | 13.4 Low | 12.0 Low | 11.4 Low |
| B | **67.6 High** | 16.1 Low | 10.2 Low | 10.3 Low | 12.0 Low | 9.9 Low | 10.1 Low |

* **Padding.** The LSTM's padded windows were never seen in training. For
  production and A, the LSTM's `risk_score` with 1–2 earlier rows is 97–99.
  That drives production's Critical result and A's High results.
* **B has no sequence input**, but its DNN was trained only on rows with 10 or
  more earlier transactions. Its first-transaction features are unseen (every
  flag is new, and `category_is_unusual` = 1).
* **Every seeded customer has 62 or more transactions**, so none of the smoke
  tests exercised this.
* **Missing history** does not crash anything:
  * `/predict` accepts an unknown customer;
  * batch scoring requires the device and location for unknown customers (as
    before).

### 3.4 Pre-existing bug: back-dated transactions are scored with the wrong row

**Symptom**
* `/predict` accepts an explicit `timestamp`.
* If that timestamp is earlier than the customer's latest known transaction,
  `build_point_features` sorts the history by time. The submitted transaction
  is then not the last row.
* `score_transaction` still scores `recomputed.iloc[-1]`, which is a different
  transaction. The similarity score is computed from that row too.

**Reproduction with production**
* Scored alone, the ₹85,000 Lagos transaction at 03:00 gives 99.9, Critical.
* After a 13:30 transaction on the same day, it gives **1.95, Low**. The row
  actually scored is the ₹3,800 Pune transaction.

**Scope**
* It affects every model set.
* The frontend and batch scoring never send timestamps (they default to now),
  so they are not affected. API clients that back-date or replay transactions
  are.
* The earlier smoke tests missed it because each one scored a single
  transaction. The end-to-end run in this audit hit it.

It was not fixed (decision D8).

### 3.5 Home device, the 90-day rule and defaults

* The home device is not a model input. `is_new_device` means "never seen in
  this customer's history".
* The profile and batch defaults use the most frequent device over the whole
  history. The 90-day rule designed in 4C-2e-a is not implemented.
* On the live v1 history both rules give the same home device for all 500
  customers. On v2 they differ for 32 customers. This matters only if the live
  history is switched to v2.

### 3.6 Conclusion

The candidates can technically score real production transactions: there are
no errors, and the features are compatible. But their behaviour on the live
customers differs markedly from the v2 evaluation setting:

* much higher alert volume under the bands;
* cold-start false alerts;
* an existing mis-scoring bug for back-dated input.

The v1 seed history used in the smoke tests hides the cold-start problem
(every customer has 62 or more transactions) and hid the back-dating bug.

## 4. The risk_score contract

The API field and its type are unchanged for every model set (float, 0–100).

| Model set | What `risk_score` is |
|---|---|
| production | LSTM probability × 100 from the 10 earlier transactions (the v1 LSTM) |
| A | the same definition, from A's LSTM |
| B | **equal to `fraud_probability`**. There is no LSTM, so it is a compatibility placeholder, not a separate signal. |

Every place that describes the risk score or LSTM-derived risk:

| Place | Current text | Correct for B? |
|---|---|---|
| `frontend/src/App.jsx:207–210` | gauge "Risk Score", caption "Trajectory risk from this customer's prior activity, before this transaction" | **No** |
| `frontend/src/App.jsx:215` | Fraud Probability caption "…can be high even if prior trajectory was clean" | **No** (it implies a separate trajectory signal) |
| `frontend/src/components/RiskGauge.jsx:1–10` | colours `risk_score` with the 25/50/80 stops | Misleading for all sets: those are probability bands |
| `frontend/src/components/FraudEvolutionTimeline.jsx:16,40` | a "Risk Score" line next to fraud probability | For B, the two lines coincide |
| `frontend/src/components/BatchScoring.jsx:104,116` | a "Risk Score" column | For B, it duplicates Fraud Probability |
| `frontend/src/components/ModelPerformance.jsx:100–106` | "LSTM Risk Predictor" block; DNN described as using "the LSTM risk score" | **No** (and shows v1 metrics for every set) |
| `backend/app/report.py:85–86` | "Risk Score … Trajectory risk from prior activity, before this transaction" | **No** |
| `backend/app/report.py:132` | footer "…outputs from the LSTM risk predictor and DNN fraud classifier…" | **No** |
| `backend/app/schemas.py:33` | OpenAPI: `risk_score` "0-100, from LSTM behavioral risk model" | **No** |
| `backend/app/db/models.py:22` | comment "0-100, from LSTM" | **No** |
| `backend/app/models/shap_explainer.py:28` | display name "Elevated Behavioral Risk Score" | Yes: B never emits `risk_score` as a reason |
| `backend/app/main.py:221` | `/metrics` docstring "…of the LSTM and DNN…" | **No** |
| `backend/app/inference_pipeline.py:8–25` | the module docstring documents the B behaviour | Yes |
| `README.md:7–11, 41, 45, 309–320` | the architecture is described as LSTM → DNN | It does not mention model sets |

Required before B is deployed (decision D5): each surface either reads the
model set or uses neutral wording. Two further points need a decision:

* Nothing in `/predict` or `/health` tells the frontend which model set is
  loaded. Showing different captions needs either an additive endpoint or
  field (for example `GET /model-set`), or neutral captions for everyone.
* Keeping `risk_score == fraud_probability` for B is the decided compatibility
  behaviour. Whether the UI should hide or relabel the gauge for B is a
  product decision.

## 5. PDF: exact changes needed (not made)

`build_pdf_report(prediction)` receives only the posted `PredictionResponse`,
so it cannot know the model set.

1. In `main.get_pdf_report`, pass the loaded model set:
   `build_pdf_report(prediction.model_dump(), model_set=get_pipeline().model_set.name)`.
   Alternatively, pass a small description dict: name, `uses_lstm`, the
   manifest weights SHA-256 and the dataset version.
2. In `report.py`, choose the wording by model set:

| Item | production (unchanged) | v2_dnn_lstm | v2_dnn_only |
|---|---|---|---|
| Risk Score row, "Meaning" column | "Trajectory risk from prior activity, before this transaction" | "LSTM risk from the customer's 10 previous transactions (input to the DNN)" | Either drop the row, or show "Same as Fraud Probability (this model set has no LSTM trajectory model)" |
| Footer | the current text | "Risk Score is the LSTM risk predictor's output and Fraud Probability the DNN classifier's output (model set v2_dnn_lstm, trained on synthetic dataset v2); reasons are SHAP attributions on the DNN." | "Fraud Probability is the output of a DNN on 9 behavioural features (model set v2_dnn_only, trained on synthetic dataset v2). This model set has no LSTM; Risk Score repeats Fraud Probability for API compatibility. Reasons are SHAP attributions on the DNN." |
| Model identification line (new, optional) | none, or "production (models/saved)" | "v2_dnn_lstm, DNN weights <sha12>, LSTM weights <sha12>" | "v2_dnn_only, DNN weights <sha12>" |

3. Add tests: the PDF text per model set, asserting that the B report contains
   neither "LSTM" nor "Trajectory".

**Caveat.** `/report/pdf` renders a prediction the client posts back. After a
switch, a PDF of an older prediction would use the new model set's wording.
Fixing that properly needs the prediction to carry its model set, which is an
additive field in the API response (decision D6).

**Today.** In the end-to-end run, the PDFs for all three model sets contained
"Trajectory risk" and "LSTM".

## 6. /metrics and /metrics/report

**Current behaviour**
* `/metrics` returns `{lstm_risk_predictor, dnn_fraud_classifier}` from
  `models/evaluation/evaluation_report.json`. That is the v1 time-split
  evaluation, `report_version` 1, dataset SHA-256 8f6977dd….
* That report evaluates evaluation copies retrained on the v1 time split, not
  the production weights in `models/saved/`.
* Its thresholds are 0.542 for the DNN and 0.998 for the LSTM. `/predict` uses
  neither; it uses the 25/50/80 bands.
* `/metrics` and `/metrics/report` return this identical content for every
  model set. This was checked on the live server for all three.

**Required for a candidate deployment** (documented, not implemented):

1. **Identify the model set:** name, directory, and for candidates the DNN and
   LSTM weights SHA-256 from the manifest. For production, the file SHA-256 of
   `models/saved/`.
2. **Identify the model version and training recipe:** the manifest
   `recipe`, seed and training data (`dataset.version`, `dataset.sha256`,
   split SHA-256).
3. **Identify the evaluation source:**
   * production → the v1 report (and state that it covers evaluation copies);
   * A and B → `models/evaluation/v2/evaluation_report.json` and
     `models/candidates/v2/comparison.json`. Both are currently local-only
     (decision D7).
4. **Identify the alert configuration:** the bands actually applied by
   `/predict` (25/50/80 today) and any manifest thresholds, so the page does
   not suggest the evaluation threshold is what triggers alerts.
5. **Frontend compatibility:** keep the two top-level keys. For B,
   `lstm_risk_predictor` should be absent or null. `ModelBlock` already renders
   nothing for missing data, so the frontend still works, but the DNN
   subtitle text needs the §4 change.

## 7. Alert thresholds

The same fixed bands (Low < 25, Medium < 50, High < 80, Critical ≥ 80, on the
capped DNN score) apply to every model set.

| | v2 test period (21,142 rows, 61 fraud) | | Live v1 history (88,913 windows) |
|---|---|---|---|
| | High+Critical per 1,000 (of which legitimate) | share of fraud rows in High+Critical | High+Critical per 1,000 (of which legitimate) |
| production | 55.7 (54.4) | 47.5% | 10.8 (1.6, in-sample) |
| A | 65.3 (63.1) | 73.8% | 32.6 (23.3) |
| B | 81.3 (79.2) | 72.1% | 28.0 (18.8) |

The manifests hold thresholds chosen on validation data:

| Candidate | F1 threshold | Alerts per 1,000, v2 test | Alerts per 1,000, live v1 |
|---|---|---|---|
| A | 0.9593 | 2.32 | 5.15 |
| B | 0.7698 | 9.46 | 8.98 |

The manifests also list FPR operating points at 0.1%, 1% and 5% validation
FPR. See the 4C-2e-c document for their test numbers and the note on
`threshold_for_fpr` ties.

**Findings**

* Under the bands, either candidate would produce tens of legitimate
  High+Critical alerts per 1,000 transactions, on both datasets.
* The bands and the validation thresholds disagree strongly:
  * A's F1 threshold (0.959) sits inside Critical, so most of A's
    High+Critical rows would not be alerts at that operating point;
  * B's threshold (0.770) sits inside High.
* The scores are uncalibrated (class-weighted training). The `%` display
  suggests a probability it does not have.
* So **model-specific thresholds, or a model-specific banding, are needed
  before either candidate is used for alerting**.
* The choice of operating point is not made here (decision D4). Options:
  * keep the bands;
  * map bands to each manifest's operating points;
  * add a separate alert flag while keeping the bands for display;
  * calibrate the scores.

## 8. Model evaluation evidence (not a ranking)

All numbers come from 4C-2d and 4C-2e-c: the v2 time split, the test period,
and validation-chosen F1 thresholds.

| | A (DNN+LSTM) | B (DNN only) |
|---|---|---|
| Test PR-AUC | 0.104 | 0.128 |
| First fraud of an episode detected | 1 of 14 | 5 of 14 |
| False positives at the F1 threshold | 40 | 176 |
| Fraud rows detected by this candidate only | none | some |

**Uncertainty**
* The PR-AUC difference (A − B) on the time split is −0.024, with a 95%
  bootstrap CI of −0.114 to +0.034.
* On the customer-grouped split it is +0.046 (CI −0.003 to +0.131).
* The sign is not consistent, so **a contribution from the LSTM is not
  demonstrated**.

**Other observations**
* At matched validation false-positive rates, the recall of A and B is close.
  The gap at the F1 thresholds mostly reflects different alert volumes.
* 39 of A's 40 false positives are legitimate foreign travel.

**Limitations**

* **Small test set.** 61 fraud transactions in 14 episodes. There are no fraud
  rings in the test period, and one warning-period episode. One episode more
  or less changes episode-level recall by about 7 percentage points.
* **Synthetic data.** It comes from a generator designed in this project, so
  the patterns the models learn are the patterns that were generated.
  Performance on real transactions is unknown.
* **One training seed** (42). Runs are reproducible, but variance across seeds
  was not measured.
* **Thresholds from a small validation period** (28 episodes). The scores are
  uncalibrated.
* **The live history is v1**, while the candidates were trained and evaluated
  on v2. No labelled evaluation exists for candidate scores on the live
  customers (§3.2 is alert volume only).
* **Production itself** was trained with the legacy random split and in-sample
  stacking, and on v1 features computed with the older code (§3.1). The v1
  numbers in `/metrics` describe evaluation copies, not the deployed weights.

## 9. End-to-end checks: frontend, API, database, batch, PDF, SHAP

**Setup.** A live uvicorn server and the Vite frontend were run, driven by
Playwright, in 5 phases that shared one SQLite database:

1. production;
2. `v2_dnn_lstm`;
3. `v2_dnn_only`;
4. rollback to production on the same database;
5. production on a fresh database.

Each phase exercised:

* `/predict` with a normal and a suspicious transaction;
* `/report/pdf`;
* a 3-row batch, including one row with blank device and location;
* `/health`, `/customers`, the profile, `/fraud-rings`, `/metrics` and
  `/metrics/report`.

Phases 1–3 also used the UI:

* a "Suspicious pattern" scan;
* the Batch Scoring tab;
* the Model Performance tab.

**Results**

| Check | production | A | B |
|---|---|---|---|
| `/predict` | 200, 14 fields | 200, 14 fields | 200, 14 fields |
| PDF | 200 (it contains "Trajectory risk" and "LSTM") | 200 (the same) | 200 (the same: **wrong for B**) |
| Batch | 200, 3 rows, summary correct | 200 | 200 (`risk_score` = `fraud_probability` on every row) |
| Other endpoints | all 200 | all 200 | all 200 |
| UI scan and batch tab | rendered, 3 batch rows | rendered | rendered: the Risk Score gauge shows 90 with the trajectory caption |
| SHAP reasons | model inputs only | may include `risk_score` ("Elevated Behavioral Risk Score"), which is correct for A | only the 9 features |
| Startup | 4.6 s | 5.2 s | 4.6 s |
| Restored from the database | 0 | 9 | 18 |

Rollback (phase 4) restored all 27 rows.

**Browser console.** The only error was one
`Failed to load resource: net::ERR_TUNNEL_CONNECTION_FAILED` per page. It is a
blocked external resource (fonts) in the sandbox and is the same for every
model set.

**Invalid configuration.** `v2_DNN_ONLY` and the empty string both made
startup fail with exit code 3 and a clear message.

**Database**
* The schema is unchanged, and rows persist and restore for every model set.
* Restoring replays only the raw transaction columns. Stored scores are never
  model inputs, so scores written by a different model set do not affect
  later scoring.
* However, no column records the model set. After a switch or rollback, the
  customer timeline and history mix scores from different model sets without
  labels. In phase 4, the production timeline showed B's 89.88/89.88 rows next
  to production's 23.89/1.78 rows.
* Whether to add provenance columns (model set, model version) is decision
  D6. That would be a schema change.

**Automated tests.** The 60 model-set tests from 4C-2d cover:

* API field and type compatibility;
* persistence, history, PDF and batch;
* SHAP alignment;
* production protection and switching.

## 10. Performance

Each model set ran in a separate process on a 2-core CPU:

* 116 scorings of seeded customers, alternating normal and suspicious;
* the first 4 excluded as warm-up;
* two runs each, reported as run 1 / run 2.

| | Pipeline load | Peak RSS after load | Median latency | p95 latency |
|---|---|---|---|---|
| production | 0.69 / 0.61 s | 760 MB | 278 / 269 ms | 391 / 374 ms |
| A | 0.63 / 0.67 s | 761 MB | 265 / 262 ms | 357 / 351 ms |
| B | 0.51 / 0.52 s | 756 MB | 203 / 213 ms | 288 / 299 ms |

* **Where the time goes.** TensorFlow and the application imports account for
  about 708 MB of memory. Latency is dominated by recomputing the customer's
  full feature history (a documented O(n) simplification) and by SHAP, not by
  the models.
* **SHAP frequency.** SHAP ran on 97–100% of these scorings for every model
  set, because nearly all of them reached the 5% display threshold.
* **Result.** No model set is slower than production. Performance is not a
  blocker.

## 11. Decisions for the project owner

These decisions have not been made in this audit.

* **D1. Keep the production model?** Production stays the default unless you
  decide otherwise. Its known issues are the legacy random-split training,
  in-sample stacking, and v1 features computed with older code.
* **D2. Deploy `v2_dnn_lstm` (A)?** If yes, first resolve D4–D8. See §8 for
  the evidence and its limits, including that the LSTM's contribution is not
  demonstrated.
* **D3. Deploy `v2_dnn_only` (B)?** If yes, first resolve D4–D8. In addition,
  the §4 wording must be fixed, because the UI and PDF otherwise describe a
  non-existent trajectory signal.
* **D4. Threshold policy.** Choose one:
  * keep the 25/50/80 bands for all model sets;
  * use per-model thresholds from the manifests (F1 or an FPR operating
    point);
  * use per-model bands;
  * calibrate the scores first.

  Also decide whether alerting and the displayed percentage should be
  separated.
* **D5. Update the PDF, metrics and UI wording before deployment?** This
  covers §5 (PDF), §6 (`/metrics`) and §4 (frontend, OpenAPI, README). It
  includes whether to add an additive way for clients to learn the loaded
  model set (for example `GET /model-set`, or a field in `/health`).
* **D6. Provenance.** Should predictions and stored rows record the model set
  and model version? This would be an additive response field and a database
  column. It is needed for correct PDFs of older predictions and for an
  unmixed timeline after a switch or rollback.
* **D7. Artifact distribution.** Should `models/candidates/v2/` (732 KB) and
  `models/evaluation/v2/` (8.5 MB) be committed, or shipped some other way?
  Today they exist only on your machine.
* **D8. Fix the known issues before any switch?** These are:
  * the back-dated transaction scoring bug (§3.4, which affects production
    too);
  * the cold-start policy (§3.3): padding in training, a minimum history, or a
    low-confidence flag;
  * whether to rebuild the v1 seed feature CSV with the current feature code
    (§3.1). This changes production's training and evaluation inputs, so it is
    a production decision too.
* **D9. Live history.** Keep scoring against the v1 seed history, or switch
  the live dataset to v2? Switching the dataset is separate from switching
  models. It would change every customer, the ring detector and the
  home-device rule (§3.5).

## 12. Verification

* **Tests.** The full backend suite passed: 317 of 317, with no skips. There
  are no code changes in this step, so the suite is the 4C-2e-d code.
* **`git diff --check`.** Clean. The trailing-whitespace check on untracked
  files is also clean.
* **Production model SHA-256** of the files in `backend/models/saved/`,
  identical before and after this audit, in the cloud copy and on the user's
  machine:

  | File | SHA-256 |
  |---|---|
  | `dnn_feature_mean.npy` | `cb12c7e19545e838c185c3e3684f601c7ec3fdbd387add15d7d7f907befa75a8` |
  | `dnn_feature_std.npy` | `5f526f065ae0d635eecbabb0b9f636fc4dc63eed778ccb4a8e738fa5aacab155` |
  | `dnn_fraud_model.keras` | `4fd3ab2e190f7c5073d7d100c7f68114babf89b02b321e72a8fb4ebb3c6d3add` |
  | `lstm_feature_mean.npy` | `cd7d3fda6554f7f027d10faf5755a01a9fe2a7d0a67eb5a748d72682a3bcb7e6` |
  | `lstm_feature_std.npy` | `86b57e2ce8b31e366397d3a5b98cb9fbf44d07ecfabf717ad4647e539e927b34` |
  | `lstm_risk_model.keras` | `81c2ab738b6d109fa3ba94f4c8fa5f7303d6f8d114f7ec892e4f5f76abb3f588` |
  | `shap_background.npy` | `ac8ffe3b53434b13880af6a7c76b1746d5a9828701af04be14c54cef697b1519` |

* **Git.** Nothing was committed or pushed.
