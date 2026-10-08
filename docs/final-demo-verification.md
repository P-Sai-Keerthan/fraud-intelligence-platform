# Final demo verification

This file describes **the current build only**: commit `9e7f265` on `main`
plus the uncommitted working-tree changes listed in section 1. Every statement
below was checked against the code in this repository or against a stored
evaluation artifact of this build on 8 October 2026. Nothing was retrained,
no threshold was changed, and no candidate model was promoted.

## 1. Build

| Item | Value |
|---|---|
| Git commit | `9e7f265` "Finalize fraud intelligence platform and review UI" (= `origin/main`) |
| Uncommitted changes | `frontend/src/api.js` (batch request timeout fix, from before this pass) and the wording changes of this pass (see the final summary of the demo-readiness pass). Nothing was committed or pushed. |
| Backend tests (`python -m pytest`) | **564 passed, 0 failed, 0 skipped** (333 library warnings: Starlette deprecation and Keras optimizer-loading notices), 11 min 24 s, run on the final code of this pass |
| Frontend lint (`npx oxlint`) | 0 warnings, 0 errors (exit code 0) |
| Frontend build (`npm run build`) | succeeds (647 modules; Vite's usual "chunk larger than 500 kB" notice only) |

## 2. Production model status

| | |
|---|---|
| Loaded by default | `production` model set, version `production-f42f2a4f6d1d`, files in `backend/models/saved/` (dataset v1) |
| How it is chosen | `MODEL_SET` environment variable, read once at start-up. Unset = `production`. An unknown value stops the server. |
| `/model-info` on a clean start | `model_set: production`, `status: PRODUCTION`, `architecture: DNN + LSTM`, `uses_lstm: true` |
| Seed-14 (`v2_dnn_lstm_seed14`) | Evaluation only. Not deployed, not promoted. Loads only if `MODEL_SET=v2_dnn_lstm_seed14` is set by hand, after a SHA-256 check of its files. Do **not** set it for the demo. |
| Other sets | `v2_dnn_lstm`, `v2_dnn_only` (seed 42): evaluation only |

## 3. What the LSTM does (checked in the code)

| | Finding | Where |
|---|---|---|
| A. Input | The customer's previous transactions as a sequence; the window ends **before** the transaction being scored | `app/models/lstm_model.py`, `app/inference_pipeline.py` |
| B. Length | 10 transactions | `SEQUENCE_LENGTH` |
| C. Features per step | The 9 behavioral features, standardized: `amount_zscore`, `amount_pct_of_avg`, `hour_is_unusual`, `is_new_device`, `is_new_location`, `is_foreign_location`, `category_is_unusual`, `failed_logins_24h`, `txn_velocity_1h` | `app/features/feature_engineering.py` |
| D. Output | One sigmoid value × 100 = **Risk Score** (0–100). Class-weighted training, not calibrated. | `lstm_model.py` |
| E. Use downstream | The Risk Score is the 10th input of the DNN (`dnn_input_columns` ends with `risk_score`) | `/model-info`, `dnn_model.py` |
| F. Training target | Whether the **next** transaction after the window is fraudulent. The DNN is trained on whether the current transaction is fraudulent; its output × 100, capped at 99.9, is the Fraud Score. | `lstm_model.py`, `dnn_model.py` |
| G. First-fraud detection measured? | **Yes.** v1 (production): the LSTM flags 0 of 16 first-fraud transactions in the test period (0 of 12 on the customer split); it flags an episode only after it has started (median 1.5 fraud transactions later), and the DNN performs identically without its score (`docs/EVALUATION.md`). The Model Performance page shows "First fraud of an episode caught 0.0%" for the LSTM. v2 final hold-out: first-fraud detection 0.605 for `v2_dnn_lstm` seed 14 against 0.690 for `v2_dnn_only` seed 14 (`docs/step4c3e-stage-c-final-evaluation.md`); multi-seed difference −0.110 [−0.194, −0.021] (`docs/step4c3e-multiseed-training-report.md`). | |
| H. Pre-fraud prediction proven? | **No.** Nothing in this repository shows that the LSTM predicts fraud before it happens. | |

What the evidence does support: on v2 data the LSTM + DNN design catches more
fraud transactions overall than the DNN alone (recall 0.558 against 0.419 on
the final hold-out; multi-seed difference +0.092 [+0.029, +0.157]). That is
the v2 candidate, which is not deployed.

**Wording used everywhere now:** "The LSTM captures temporal patterns and
ongoing suspicious behavior from a customer's recent transaction history" /
"produces a temporal risk signal for the downstream fraud classifier". The
LSTM was not removed or changed.

**Viva answer.** "The LSTM is used to model the customer's recent transaction
sequence. It captures temporal behavioral patterns and produces a temporal risk
signal that is used by the downstream fraud classifier. We do not claim that
the LSTM independently proves that fraud can always be predicted before it
occurs."

*Q: Does the LSTM predict fraud before it happens?* "It is designed to capture
temporal signals that may precede or accompany suspicious behavior, but our
evaluation does not justify claiming guaranteed pre-fraud prediction. Our
defensible claim is temporal behavioral risk detection."

## 4. Fraud Score

**The Fraud Score is a model score, not a calibrated probability.** It is the
DNN output × 100, capped at 99.9. Class-weighted training pushes the outputs up,
so 99.9 means "ranked among the most suspicious", not "99.9% chance of fraud".
The API field is still called `fraud_probability` for compatibility only. Alert
levels are fixed bands on this score: below 25 Low, 25 to below 50 Medium, 50 to
below 80 High, 80 and above Critical (`alert_level_from_probability`).

## 5. Where each value on screen comes from

| Value | Source |
|---|---|
| Fraud Score, Risk Score, alert level, behavioral similarity, SHAP values | Backend, per transaction (`POST /predict`) |
| Customer profile, history, score timeline | Backend (`/customer/{id}/profile`, `/history`); labelled "Derived from transaction history" |
| SHAP labels "Primary driver / Strong / Contributing", share % | Derived in the UI from the backend SHAP values (stated under the chart) |
| Fraud ring list, customers, devices, transaction counts | Backend (`GET /fraud-rings`): devices used by 2 or more customers |
| Cluster grouping and "Size-based level (UI)" | Derived in the UI from the ring list; not a model output |
| Batch summary tiles and filters | Counted in the UI from the backend batch results |
| Model Performance metrics | Stored evaluation artifacts: `evaluation_report.json` (production, v1) and `final_holdout_report.json` (candidates) |
| Header status / model set / version | Backend (`/health`, `/model-info`) |

No static value is presented as a prediction.

## 6. Dataset and evaluation limitations

- All data is synthetic. No result says anything about real bank traffic.
- The v1 data that the production model was trained and evaluated on turned
  out to be too easy; its near-perfect metrics are not evidence of real-world
  performance (`docs/EVALUATION.md`).
- v1 has only 16 test fraud episodes, so first-fraud figures move in steps of
  about 6 percentage points.
- Customers with fewer than 10 earlier transactions do not get an LSTM score;
  the DNN receives the training-average Risk Score and accuracy is weaker.
- Live Scan uses the current clock time, so "unusual hour" changes results
  through the day (`docs/final-review-demo-scenarios.md`).

## 7. Verified for this build (8 October 2026, clean database, production)

| Check | Result |
|---|---|
| Clean-database start-up with `MODEL_SET` unset | starts, creates the database, `/health` → `{"status":"ok"}` |
| `/model-info` | `production`, `production-f42f2a4f6d1d` |
| Header | ONLINE, model set production |
| Live Scan | CUST_0376, "Suspicious Pattern" quick scenario → Critical Risk; captions show "Temporal behavioral risk" and "not a calibrated probability" |
| PDF report | "Download Report" downloads a PDF; it uses the temporal wording and "not a calibrated probability" |
| Batch Scoring | 200-row mixed CSV scored, results and "Download results" shown |
| Fraud Rings | page loads with "Shared devices" and "Shared-device clusters"; 20 rings / 9 clusters / 29 customers on the seed data |
| Model Performance | loads; LSTM block shows "First fraud of an episode caught 0.0%" |
| Browser console | no application errors. Two expected messages: a 404 from `/customer/{id}/history` for a customer with no scans yet (handled by the UI, empty timeline) and a Google Fonts request that fails without internet (the page falls back to system fonts). |

## 8. Not verified

- The candidate model sets in the browser (only `production` was run).
- PostgreSQL (configurable, never tested).
- Browsers other than Chromium; screens narrower than a laptop in detail.
- Behavior outside 09:00–18:59 for the scenario table.
- Performance under concurrent users.
- Anything on real (non-synthetic) data.

## 9. Reports and PDFs

| File | Status |
|---|---|
| `docs/final-report/Fraud_Intelligence_Platform_Technical_Report.{docx,pdf}` | Regenerated for this build (32 pages): temporal LSTM wording, no pre-fraud claim, screenshots retaken from the current UI |
| `docs/Fraud_Intelligence_Platform_5-6_Page_Overview.{docx,pdf}` | Regenerated for this build (6 pages) |
| `docs/Fraud_Intelligence_Platform_Feature_Guide_and_Test_Examples.pdf` | Regenerated for this build (12 pages); Live Scan screenshot retaken |
| PDF produced by the app (`POST /report/pdf`) | Generated live; checked in section 7 |
| `docs/archive/Project_Review_2026-07_OBSOLETE.docx` | **Obsolete** (July 2026, older build; uses "Fraud Probability %" and a "flag risk before" claim). Moved out of `docs/`; do not show it. |

Numbers in these documents are copied from the stored evaluation reports; none
were estimated.

## 10. Known limitations

- Fraud Score is not calibrated (section 4).
- The LSTM does not detect the first fraud of an episode and is not shown to
  predict fraud before it happens (section 3).
- The production model was trained on the easy v1 data; the better v2
  candidate (seed 14) is not deployed because promotion is blocked by open
  items (`docs/step4c3f-controlled-promotion-preparation.md`).
- Fraud rings are a rule over shared device IDs, not a model; a shared device
  is a lead for investigation, not proof.
- Every Live Scan is stored and changes that customer's later scores.
- No authentication; single-user prototype.

## 11. Start-up commands (Windows, PowerShell)

Terminal 1 — backend:

```powershell
cd C:\Users\HP\Desktop\fraud-intelligence-platform\backend
.\venv\Scripts\Activate.ps1
Remove-Item Env:MODEL_SET -ErrorAction SilentlyContinue
Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for "Application startup complete", then check http://localhost:8000/health
shows `{"status":"ok"}`.

Terminal 2 — frontend:

```powershell
cd C:\Users\HP\Desktop\fraud-intelligence-platform\frontend
npm run dev
```

Open http://localhost:5173. The header must show ONLINE and model set
`production`.

Clean database (optional, recommended): stop the backend, rename
`backend\fraud_platform.db` (for example to `fraud_platform_before_demo.db`),
and start the backend again; it creates a new, empty database.

## 12. Demo sequence (about 10 minutes)

1. **Header** — ONLINE, model set production. Say: all data is synthetic.
2. **Live Scan, normal** — customer for the current hour from
   `docs/final-review-demo-scenarios.md`, scenario A → Low Risk. Point out
   Fraud Score (a model score, not a probability) and the Risk Score (temporal
   risk signal from the previous 10 transactions).
3. **Live Scan, suspicious** — scenario C → Critical. Walk through the SHAP
   reasons; say SHAP explains the score, it does not prove fraud.
4. **Download Report** — open the PDF.
5. **Batch Scoring** — upload `test_data/batch_scoring/batch_mixed_transactions.csv`, show
   the alert distribution, filter Critical, download results.
6. **Fraud Rings** — open cluster `DEV_UNKNOWN_1125 +5`: 7 customers, 6 shared
   devices. Say it is a shared-device rule and a lead, not proof.
7. **Model Performance** — production metrics are from the v1 evaluation;
   show the LSTM's 0.0% first-fraud figure and explain why we do not claim
   pre-fraud prediction. Mention Seed-14 as evaluated, not deployed.
