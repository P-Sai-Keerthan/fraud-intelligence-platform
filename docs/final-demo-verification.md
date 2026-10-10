# Final demo verification

This file describes **the current build only**: commit `33565e3` on `main`
plus the uncommitted Step 4D changes (the downstream classifier was replaced:
LSTM → random forest). Every statement below was checked against the code in
this repository or against an evaluation artifact of this build on 8 October
2026. Nothing has been committed or pushed.

## 1. Build

| Item | Value |
|---|---|
| Git commit | `33565e3` "Finalize demo-ready build and align documentation", plus uncommitted Step 4D changes |
| Backend tests (`python -m pytest`) | **615 passed, 0 failed, 0 skipped** (332 library warnings), 17 min 48 s, on the final code |
| Frontend lint (`npx oxlint`) | 0 warnings, 0 errors (exit code 0) |
| Frontend build (`npm run build`) | succeeds (Vite's usual "chunk larger than 500 kB" notice only) |

## 2. Model status

| | |
|---|---|
| Loaded by default | `v2_lstm_rf_seed14`, version `v2_lstm_rf_seed14-483c1b395789`, from `backend/models/candidates_downstream/v2/seed_14/lstm_random_forest/` |
| Architecture | `LSTM + Random Forest`: the seed-14 v2 LSTM, unchanged (byte-identical copy, weight hash checked), followed by a random forest. The forest has 200 trees, max depth 12, at least 5 samples per leaf and no class weighting. |
| How it was chosen | Pre-registered protocol `docs/step4d-downstream-selection-protocol.md`. Four classifiers were compared on development data (DNN, logistic regression, random forest, gradient boosting), and the result was confirmed once on a fresh hold-out. Full report: `docs/model_selection_report.md`. |
| How it is chosen at start-up | `MODEL_SET` environment variable, read once. Unset = `v2_lstm_rf_seed14`. An unknown value stops the server. Every file must match its pinned SHA-256 before it is loaded. |
| `/model-info` on a clean start | `model_set: v2_lstm_rf_seed14`, `status: PRODUCTION`, `architecture: LSTM + Random Forest`, `training_seed: 14`, `uses_lstm: true`, `calibrated_probabilities: false` |
| Previous default | `MODEL_SET=production` (v1 LSTM → DNN, `backend/models/saved/`). Unchanged and loadable; labelled "PREVIOUS DEFAULT (v1)" |
| Other sets | `v2_dnn_lstm_seed14`, `v2_dnn_lstm`, `v2_dnn_only`: evaluation only |

## 3. What the LSTM does (unchanged)

* **Input:** the customer's previous 10 transactions. Each transaction has the
  9 behavioral features, standardized. The window ends before the transaction
  being scored.
* **Output:** a sigmoid value × 100, the Risk Score. It is an input to the
  random forest.
* **Training target:** whether the next transaction after the window is
  fraudulent.
* **Not changed by Step 4D.** The classifier was retrained on the LSTM's
  out-of-fold scores, recomputed bit for bit (`reproduction.json`).
* **No pre-fraud claim.** It is not shown to predict fraud before it happens:
  on v1 its own first-fraud recall is 0/16 (`docs/EVALUATION.md`).
* **Wording:** "The LSTM captures temporal patterns and ongoing suspicious
  behavior from a customer's recent transaction history and produces a
  temporal risk signal."

## 4. Fraud Score

**The Fraud Score is a model score, not a calibrated probability.** It is the
random forest's output × 100, capped at 99.9.

* On the fresh hold-out the score agrees with the observed fraud rate below
  about 10.
* Above about 30 it understates it: scores of 30–40 are fraud 58.6% of the
  time.
* The API field keeps its historical name `fraud_probability`.
* **Alert bands** are the fixed bands, unchanged: below 25 Low, 25 to below 50
  Medium, 50 to below 80 High, 80 and above Critical.
* **The bands were designed for the earlier DNN.** The random forest's
  validated alert cut-off is a score of 4.39 and its Critical cut-off is
  26.23. These are what every reported metric uses; they are not applied to
  the live bands.

## 5. Why the old DNN showed about 100% accuracy

* **The v1 data is too easy.** On the v1 test period one feature alone
  (`hour_is_unusual`) separates fraud with precision 1.000 and recall 1.000.
* **Class imbalance.** Predicting "legitimate" for everything already scores
  0.9894 accuracy.
* **No leakage was found** (`audit.json`; report, section 3).
* **Accuracy cannot separate models.** On v2 every candidate's accuracy is
  0.988 to 0.989, while PR-AUC ranges from 0.293 to 0.471.

## 6. Headline results (synthetic data; fresh hold-out, scored once)

Customers with at least 10 earlier transactions, each model at its own
validation cut-off:

| Model | PR-AUC | Recall | Precision | Legit alerts / 1,000 | First fraud detected |
|---|---|---|---|---|---|
| **LSTM + random forest (default)** | **0.464** | **0.645** | 0.236 | 10.23 | 0.738 |
| LSTM + DNN (same LSTM) | 0.328 | 0.548 | 0.217 | 9.68 | 0.595 |
| Previous default (v1) | 0.133 | 0.255 | 0.104 | 10.71 | 0.384 |

Development data, mean of 5 training seeds, PR-AUC:

| DNN | Logistic regression | Random forest | Hist. gradient boosting |
|---|---|---|---|
| 0.293 | 0.381 | 0.471 | 0.388 |

## 7. Where each value on screen comes from

| Value | Source |
|---|---|
| Fraud Score, Risk Score, alert level, behavioral similarity, SHAP values | Backend, per transaction (`POST /predict`). SHAP is exact Tree SHAP of the same random forest that scored. |
| Customer profile, history, score timeline | Backend (`/customer/{id}/profile`, `/history`) |
| SHAP labels "Primary driver / Strong / Contributing", share % | Derived in the UI from the backend SHAP values |
| Fraud ring list | Backend (`GET /fraud-rings`): devices used by 2 or more customers |
| Cluster grouping and "Size-based level (UI)" | Derived in the UI |
| Model Performance: comparison table, hold-out table, gates, calibration, new-customer card | Backend `/metrics` → `downstream_evaluation`, copied from `backend/models/evaluation/downstream/*.json` |
| Header model set / version / PRODUCTION chip | Backend (`/model-info`) |

## 8. Verified for this build (8 October 2026, clean database, `MODEL_SET` unset)

| Check | Result |
|---|---|
| Clean-database start-up | starts; `/health` → `{"status":"ok"}`; `/model-info` as in section 2 |
| Header | ONLINE, model set `v2_lstm_rf_seed14`, PRODUCTION |
| Live Scan | CUST_0376, "Suspicious Pattern" (Rs. 75,000, electronics, DEV_UNKNOWN_9999, Lagos, 4 failed logins) → Critical Risk, Fraud Score 80.8, Risk Score 15. SHAP reasons: Foreign Location, New Device, Amount Far Above Average, Multiple Failed Logins. |
| PDF report | downloads; says "LSTM risk score -> Random Forest", "random forest", "Tree SHAP", "not a calibrated probability" |
| Batch Scoring | 30-row mixed CSV scored, results and download shown |
| Fraud Rings | page loads ("Shared devices", "Shared-device clusters") |
| Model Performance | shows the classifier comparison, the fresh hold-out confirmation (Confirmed), calibration table and the new-customer limitation |
| Hand-built scenarios A–H and five normal purchases | `docs/final-review-demo-scenarios.md`, last section: normal purchases score about 0.06; single signals 0.2–16; failed-login attack 43.4; several signals 74.6 |
| Browser console | no application errors. Two expected messages: a 404 from `/customer/{id}/history` for a customer with no scans yet (handled), and a Google Fonts request that fails without internet. |

## 9. Not verified

* Browsers other than Chromium.
* PostgreSQL (configurable, never tested).
* Performance under concurrent users.
* Anything on real (non-synthetic) data.
* XGBoost: not installed, not evaluated.

## 10. Reports and PDFs

| File | Status |
|---|---|
| `docs/model_selection_report.md` | **Current.** Classifier selection, audit, metrics, calibration, SHAP, limitations, reproduction |
| `docs/step4d-downstream-selection-protocol.md` | Current: the pre-registered rule |
| PDF produced by the app (`POST /report/pdf`) | Current; checked in section 8 |
| `docs/final-report/*`, `docs/Fraud_Intelligence_Platform_5-6_Page_Overview.*`, `docs/Fraud_Intelligence_Platform_Feature_Guide_and_Test_Examples.pdf` | **Out of date for the classifier.** They describe the DNN-based build before Step 4D, including its screenshots and scores. Do not present their model sections as the current system; they need regenerating. |
| `docs/final-review-*.md` | Each starts with a Step 4D note saying what still applies |
| `docs/archive/Project_Review_2026-07_OBSOLETE.docx` | Obsolete |

## 11. Known limitations

1. All data is synthetic. No result describes real banking performance.
2. The Fraud Score is not a calibrated probability.
3. The fixed alert bands do not suit the random forest's lower scores. On the
   hold-out, 62.1% of fraud transactions fall in "Low Risk" under the bands;
   the evaluated cut-off is 4.39. Deriving bands for this model is an owner's
   decision.
4. Customers with fewer than 10 earlier transactions: 59.4 legitimate alerts
   per 1,000, against 25.0 for the DNN. No cold-start policy has been
   approved.
5. The random forest raises about 0.5 more legitimate alerts per 1,000 than
   the DNN for established customers (within the pre-registered limit).
6. The app's customer histories are v1 seed data, while the model was trained
   on v2 data.
7. The LSTM does not predict fraud before it happens.
8. Fraud rings are a shared-device rule, not a model.

## 12. Start-up commands (Windows, PowerShell)

Terminal 1, backend:

```powershell
cd C:\Users\HP\Desktop\fraud-intelligence-platform\backend
.\venv\Scripts\Activate.ps1
Remove-Item Env:MODEL_SET -ErrorAction SilentlyContinue
Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for "Application startup complete", then check that
http://localhost:8000/health shows `{"status":"ok"}`. The log line must say
`model set 'v2_lstm_rf_seed14'` and `LSTM risk score -> random forest`.

Terminal 2, frontend:

```powershell
cd C:\Users\HP\Desktop\fraud-intelligence-platform\frontend
npm run dev
```

Open http://localhost:5173. The header must show ONLINE, model set
`v2_lstm_rf_seed14` and PRODUCTION.

**Clean database (recommended):** stop the backend, rename
`backend\fraud_platform.db`, then start the backend again.

**Previous default:** run `$env:MODEL_SET = "production"` before starting the
backend.

## 13. Demo sequence (about 10 minutes)

1. **Header.** Point out the model set and PRODUCTION. Say: all data is
   synthetic.
2. **Live Scan, normal.** Use the customer's usual amount, category, device,
   city and hour. The result is Low Risk, Fraud Score near 0. Explain the
   Risk Score (temporal risk signal from the previous 10 transactions) and the
   Fraud Score (model score, not a probability).
3. **Live Scan, suspicious.** Use "Suspicious Pattern" → Critical. Walk
   through the SHAP reasons: exact contributions of the same random forest,
   which explain the score but do not prove fraud.
4. **Download Report.**
5. **Batch Scoring.** Upload
   `test_data/batch_scoring/batch_mixed_transactions.csv`.
6. **Fraud Rings.** A shared-device rule; a lead, not proof.
7. **Model Performance.**
   * Why the old 100% was meaningless (an easy v1 dataset, class imbalance).
   * The four-classifier comparison: PR-AUC first, never accuracy.
   * The fresh hold-out confirmation.
   * The calibration table (not a probability).
   * The new-customer limitation.
