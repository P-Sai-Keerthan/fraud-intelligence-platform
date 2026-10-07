# Explainable Fraud Intelligence Platform

An AI-powered banking fraud intelligence system that builds a **Behavioral
Fraud DNA** profile per customer, uses recent transaction sequences to
estimate temporal fraud risk (LSTM), scores each individual transaction
(DNN), and explains the score (SHAP).

> **Read this first:** all data is **synthetic**, and both model outputs are
> **risk scores, not calibrated probabilities**. See *Known limitations* below
> before quoting any number from this project.

This repository is fully working end-to-end right now, using a **synthetic
dataset** generated with realistic behavioral patterns (see
`data/generate_synthetic_data.py`). Trained models are already included
under `backend/models/saved/`, so you can run the API and dashboard
immediately without retraining anything. When you get access to a real
dataset (PaySim, IEEE-CIS Fraud Detection, or your own data), swap it in and
retrain — the pipeline doesn't change.

---

## 1. Project structure

```
fraud-intelligence-platform/
├── data/
│   ├── generate_synthetic_data.py     # generates the demo dataset
│   ├── transactions.csv                # raw synthetic transactions (generated)
│   ├── transactions_with_features.csv  # + behavioral features (generated)
│   └── lstm_X.npy, lstm_y.npy, lstm_meta.csv   # LSTM training sequences (generated)
│
├── backend/
│   ├── requirements.txt
│   ├── models/saved/                   # trained model weights (already included)
│   └── app/
│       ├── config.py                    # all file paths, resolved automatically
│       ├── main.py                      # FastAPI app (run this to start the API)
│       ├── inference_pipeline.py        # orchestrates LSTM -> DNN -> SHAP -> similarity
│       ├── schemas.py                   # request/response models + input validation rules
│       ├── security.py                  # CORS allow-list, request-body size limit, in-process rate limiter
│       ├── batch.py                     # batch-CSV parsing (validate every row before scoring)
│       ├── profile.py                   # behavioral context + typical-purchase values from a customer's own history
│       ├── features/feature_engineering.py   # Behavioral Fraud DNA feature builder
│       ├── models/
│       │   ├── lstm_model.py            # LSTM risk predictor
│       │   ├── dnn_model.py              # DNN fraud classifier
│       │   ├── shap_explainer.py          # Explainable AI (SHAP) wrapper + feature-state label checks
│       │   ├── evaluate.py                # held-out metrics served by GET /metrics
│       │   ├── skew_check.py              # measures the training/inference feature skew (python -m app.models.skew_check)
│       │   └── similarity.py              # Behavioral Similarity Score
│       └── db/                          # SQLAlchemy models + session (SQLite by default)
│   └── tests/                           # Phase 1 regression tests (pytest)
│
├── frontend/                            # React + Tailwind + Recharts dashboard
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── components/
│
└── docs/
    └── RESEARCH_NOTES.md                # paper-writing guide: metrics, structure, related work
```

---

## 2. Prerequisites

Install these first if you don't have them:

- **Python 3.10 or newer** — check with `python3 --version`
- **Node.js 18 or newer** — check with `node --version`
- **VS Code** (recommended) with the Python extension installed

---

## 3. Backend setup (run these commands in order)

Open a terminal in the project root folder (`fraud-intelligence-platform/`).

### 3.1 Create and activate a virtual environment

```bash
cd backend
python3 -m venv venv
```

Activate it:

```bash
# Mac/Linux:
source venv/bin/activate

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Windows (Command Prompt):
venv\Scripts\activate.bat
```

You'll know it worked when your terminal prompt starts with `(venv)`.

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, TensorFlow, SHAP, SQLAlchemy, and everything else
needed. It may take a few minutes the first time (TensorFlow is large).

### 3.3 Start the API server

```bash
uvicorn app.main:app --reload --port 8000
```

You should see output ending with something like:
```
[pipeline] Loading trained models...
[pipeline] Loaded history for 500 customers.
INFO:     Application startup complete.
```

Leave this terminal running. Open **http://localhost:8000/docs** in your
browser — you'll see the auto-generated Swagger UI where you can test every
endpoint directly.

### 3.4 Quick test with curl (open a NEW terminal for this)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_0001",
    "amount": 85000,
    "merchant_category": "electronics",
    "device_id": "DEV_UNKNOWN_9999",
    "location": "Lagos",
    "failed_logins_24h": 4
  }'
```

You should get back a JSON response with `risk_score`, `fraud_probability`,
`alert_level: "Critical Risk"`, and a list of `reasons` like "Foreign
Location" and "New Device".

---

## 4. Frontend setup (run these commands in order)

Open a **new terminal** (keep the backend running in the other one), go to
the project root, then:

```bash
cd frontend
npm install
npm run dev
```

You'll see output like:
```
  VITE ready in 400 ms
  ➜  Local:   http://localhost:5173/
```

Open **http://localhost:5173** in your browser. Select a customer from the
dropdown (e.g. `CUST_0001`), click one of the quick scenario buttons
("Typical purchase" / "Suspicious pattern"), and click **Scan Transaction**.
You'll see the temporal risk gauge, fraud risk score, alert banner, SHAP
explanation chart, and the fraud evolution timeline update live.

The dev server automatically proxies `/api` requests to your backend on
port 8000 (configured in `vite.config.js`), so no extra setup is needed for
local development.

---

## 5. (Optional) Regenerating data and retraining models

You don't need to do this to run the project — trained models are already
included. Do this only if you want to modify the feature engineering,
model architecture, or swap in a real dataset.

Run these from inside the `backend/` folder, in this exact order:

```bash
cd backend
python -m app.features.feature_engineering   # ~60-90 seconds
python -m app.models.lstm_model                # trains the LSTM, ~3-5 minutes on CPU
python -m app.models.dnn_model                  # trains the DNN, ~1 minute
python -m app.models.shap_explainer              # sanity-check SHAP explanations
```

To regenerate the underlying synthetic dataset from scratch first (only
needed if you want a different random sample):

```bash
cd ..                    # back to project root
python3 data/generate_synthetic_data.py
cd backend
python -m app.features.feature_engineering
python -m app.models.lstm_model
python -m app.models.dnn_model
```

### Swapping in a real dataset (PaySim / IEEE-CIS)

1. Download PaySim or IEEE-CIS Fraud Detection from Kaggle.
2. Reshape it into the same columns `data/generate_synthetic_data.py`
   produces (see the schema documented at the top of that file):
   `customer_id, transaction_id, timestamp, amount, merchant_category,
   device_id, location, failed_logins_24h, is_new_device, is_new_location,
   is_fraud`.
3. Save it as `data/transactions.csv`, replacing the synthetic one.
4. Re-run the three commands in section 5 above.

Nothing else in the codebase needs to change — the feature engineering,
models, and API all work off that one schema.

---

## 6. How the system works (for your report/paper)

1. **Behavioral Fraud DNA** (`feature_engineering.py`): for every
   transaction, computes 9 behavioral features using *only* that
   customer's prior history (no lookahead) — amount z-score vs their own
   average, whether the hour/device/location/category is new or unusual,
   transaction velocity, failed logins.
2. **LSTM Temporal Risk Model** (`lstm_model.py`): takes the customer's last
   10 transactions' behavioral features (all *before* the current one) as a
   sequence and outputs a 0-100 temporal risk score, trained on "is the next
   transaction fraudulent". It detects temporal patterns associated with
   fraudulent behavior. It has **not** been shown to predict the first fraud
   event of a burst (see limitations).
3. **DNN Fraud Risk Model** (`dnn_model.py`): takes the current transaction's
   own behavioral features *plus* the LSTM risk score, and outputs a 0-100
   **Fraud Risk Score** for *this specific transaction*. The API field is still
   named `fraud_probability` for compatibility, but it is a model score, **not**
   a calibrated probability.
4. **SHAP Explainer** (`shap_explainer.py`): wraps the DNN with
   `shap.GradientExplainer` and maps the top contributing features to
   human-readable reasons ("New Device", "Foreign Location", etc.). A positive
   SHAP value only means the model's score went up because of that input, so a
   descriptive label is shown **only if the transaction actually exhibits that
   condition** (e.g. "High Transaction Velocity" requires >= 2 other
   transactions in the previous hour). Otherwise the factor is dropped, or shown
   with neutral wording if nothing anomalous remains.
5. **Behavioral Similarity Score** (`similarity.py`): z-scores the current
   transaction against the customer's own historical mean/std per feature,
   and maps the average deviation to a 0-100 similarity percentage via
   exponential decay.

### Known limitations (be upfront about these in your paper — reviewers expect it)

- The dataset is **synthetic**. Real bank data is never public, so this is
  standard practice in fraud-detection research, but say so explicitly in
  your methodology section. It is also **highly separable by construction**
  (for example every fraud row has an unusual hour and every foreign-location
  row is fraud), which is why the DNN reaches AUC 1.000 and recall 1.000 on the
  held-out split. A hand-written two-clause rule performs comparably. These
  results show the pipeline works end-to-end on the supplied data; they are
  **not** evidence of real-world fraud-detection performance.
- **Scores are not calibrated probabilities.** Class-weighted training inflates
  them (LSTM mean score ~13.6 against a ~0.9% fraud base rate) and no
  calibration method has been applied or validated.
- **The LSTM is not shown to predict fraud before it starts.** In the synthetic
  data, fraud arrives in bursts and ~93% of fraud-positive sequences already
  contain an earlier fraud transaction; it caught none of the 60 first-in-burst
  cases in an audit analysis (an analysis done outside the app, not a feature of
  it). Describe it as temporal risk estimation, not early prediction.
- **Single signals can score low.** Amount x10 or failed-logins x5 *alone* can
  score Low, while combinations of signals score High/Critical. Removing the
  inference-time input clip does not change this (it makes those scores lower),
  so it is a property of the shipped model. A likely cause: it was trained on a
  feature file generated before the z-score fix (that file still contains
  z-scores up to +/-1,585, and about half of the rows with such extreme values
  are not fraud). Confirming and fixing this needs retraining on regenerated
  features, which has not been done.
- **Accepted transactions update the customer's baseline** (by design). Input
  that is invalid (NaN/Infinity, out of bounds, malformed) is rejected before any
  state changes, but a *valid* extreme transaction still enters the history.
- The evaluation uses a random 80/20 split of overlapping sequences; no
  customer-disjoint or time-based evaluation is part of the app.
- **Training/inference feature skew (known, measured, not changed).** The shipped
  models were trained on `data/transactions_with_features.csv`, generated *before*
  the `amount_zscore` fix now in `feature_engineering.py` (std floored at 10% of the
  average, z-score clipped to +/-10; `amount_pct_of_avg` clipped to 1000). The CSV
  still holds z-scores from -1,585 to +1,163, so `amount_zscore` differs between
  the training file and what live inference computes on 43,441 of 93,913 rows.
  Additionally the DNN input is clipped to +/-6 sigma at inference (and in the
  metrics), but was not clipped in training. Measured effect on the held-out
  metrics (`python -m app.models.skew_check`): LSTM precision 0.7042 -> 0.7042,
  ROC-AUC 0.9624 -> 0.9627; DNN precision 0.8367 -> 0.8410 (one fewer false
  positive), recall 1.000 -> 1.000. The shipped model is deliberately preserved:
  correcting this properly means regenerating the features and retraining, and an
  inference-side "undo the fix" would bring back the unbounded z-scores the fix
  removed. Retraining on regenerated features is the real fix and is future work.
- **Cold-start scores have limited sensitivity.** The DNN leans on history-based
  features. For a customer with no history those inputs are unavailable (see the
  cold-start policy below), so the score rests on foreign-location / failed-login
  signals only. A "Low Risk" label for a brand-new customer therefore means "no
  behavioral evidence of risk", not "safe".
- **The dashboard loads its fonts from Google Fonts** (Space Grotesk, Inter, JetBrains Mono; see
  `frontend/index.html`), so it needs internet access to look exactly as designed. Offline it falls back
  to the system font stacks and stays fully usable. Self-hosting the font files would remove the external
  dependency; it was left as is because it is cosmetic. The PDF report writes amounts as "Rs." rather than
  the rupee sign because its built-in font has no rupee glyph.
- **Accessibility is improved but not audited against WCAG.** The dashboard has labelled form controls,
  tab semantics with keyboard navigation, text equivalents for the gauges and charts, a visible focus
  ring and AA text contrast for body and caption text. It has not been tested with a screen reader
  or an automated WCAG audit, and form-field borders are low contrast (kept, to preserve the visual design).
- **Demo-level API protection only.** There is no authentication, and the rate
  limiter is in-process and per-IP. Production would additionally need: real
  authentication/authorisation, TLS, a distributed rate limiter shared across
  workers, per-user quotas, audit logging, secrets management, and a reverse
  proxy/WAF. Do not expose this API to an untrusted network as-is.
- Behavioral features are recomputed from a customer's full history on
  every request (`O(n)` per prediction). Fine at demo scale; a production
  system would maintain incrementally-updated rolling statistics instead.
- The Behavioral Similarity Score weights all 9 features equally. It can
  diverge from the DNN's fraud risk score for customers with naturally
  tight variance in one dimension (e.g. very consistent spending amounts).
  This is a good "future work" paragraph for your paper: learned feature
  weighting for the similarity metric.
- Report **Precision, Recall, F1, ROC-AUC, PR-AUC and the False Positive Rate**
  in your results — not just accuracy, since fraud is heavily imbalanced and
  accuracy alone looks artificially high. All of these are computed by
  `GET /metrics` / the Model Performance tab, alongside the dataset caveats.

See `docs/RESEARCH_NOTES.md` for a full IEEE/Springer paper structure
template and suggested related-work citations.

---

## 6b. API behaviour worth knowing

- `POST /predict` validates every field **before** anything else runs. Rejected
  with a clear `422` (and no change to any customer history): NaN/Infinity/`1e400`,
  non-positive or > ₹1,000,000 amounts, negative or > 100 failed logins, empty or
  > 64-character ids/locations, malformed or implausible timestamps (before 2000,
  or > 7 days ahead). Timezone-aware timestamps are converted to server local time.
- A transaction is scored against the history that **precedes its own timestamp**,
  so a back-dated timestamp is handled at its correct chronological position.
- `GET /customer/{id}/profile` returns the customer's behavioral context (typical
  amount, usual device/location/hours, ...) computed only from their stored history,
  plus a *typical purchase* built from those values. The dashboard's "Typical
  purchase" button and "What changed?" panel use it.
- `POST /predict/batch` accepts up to 50 rows / 1 MB. Every row is validated first;
  if any row is invalid, nothing is scored (`422` listing the bad rows); too many
  rows returns `413`. Blank `device_id`/`location` use the customer's usual ones.
  Inference runs in a worker thread so `/health` stays responsive, and only one
  batch runs at a time (a second returns `429` with `Retry-After`).
- If saving to the database fails, the request returns `503` and the in-memory
  history is left unchanged.

Run the regression tests (real models, ~1-2 minutes):

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests -q
```

---

## 7. Troubleshooting

**"Address already in use" when starting uvicorn** — something's already
running on port 8000. Either stop it, or run on a different port:
`uvicorn app.main:app --reload --port 8001` (and update
`frontend/vite.config.js`'s proxy target to match).

**Frontend shows "Could not reach the backend"** — make sure the backend
terminal is still running and shows no errors. Visit
http://localhost:8000/health directly in your browser — it should return
`{"status": "ok"}`.

**`ModuleNotFoundError` when running training scripts** — make sure you're
in the `backend/` folder and your virtual environment is activated, and
that you're using `python -m app.models.lstm_model` (with `-m`), not
`python app/models/lstm_model.py` directly — the `-m` flag is required for
the internal relative imports to resolve correctly.

**TensorFlow prints CUDA/GPU warnings** — these are harmless. There's no
GPU available, so it falls back to CPU, which is fine for this project's
scale.

---

## 6c. Cold start, security and concurrency behaviour (Phase 2)

### Cold-start policy (customers with little or no history)

The LSTM was trained only on full 10-transaction windows of real history, and a
behavioral baseline needs enough transactions for a meaningful mean/std. The system
never manufactures history (it used to feed the LSTM an all-zero or padded window and
compare against a baseline of zeros). The tier is reported as `history_status` in the
`/predict` response and in `/customer/{id}/profile`:

| `history_status` | Prior transactions | What is computed |
|---|---|---|
| `none` | 0 (also: an unknown `customer_id`) | History-based inputs (amount z-score, unusual hour/category, new device/location, amount vs average) and the LSTM do not exist, so each is fed to the DNN at its **training mean** ("no information"). The fraud risk score rests only on genuinely available signals (foreign location, failed logins). `risk_score`, `similarity_pct`, `deviation_pct` are `null`; reasons never name a history-based factor. |
| `limited` | 1-9 | Features come from the real (short) history. `risk_score` (LSTM) and similarity are `null`: the LSTM is never run on a padded window. |
| `established` | 10 or more | Everything, exactly as before. Verified unchanged: 200 established-customer results scored by the previous code and the current code were identical in LSTM risk, DNN score, similarity, deviation and alert level. |

`response.history_transactions` is the number of prior transactions used. The dashboard
shows a "Limited behavioral history" notice and "Not available" instead of a number.
Unknown customers are still scorable (new customers must be), but they are only
registered after a **valid** request succeeds; an invalid request creates nothing.
This is the one deliberate change to the response schema: `risk_score`,
`similarity_pct`, `deviation_pct` (and the same field on batch results / timeline points)
may now be `null`, and `history_status` / `history_transactions` were added.

### Security configuration (all via real environment variables; `.env` files are not read)

| Variable | Default | Meaning |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated browser origins allowed to call the API cross-origin. A lone `*` is honoured but credentials are **never** enabled. The Vite dev server proxies `/api`, so local development needs no CORS at all. |
| `RATE_LIMIT_ENABLED` | `1` | `0` disables the limiter. |
| `RATE_LIMIT_PREDICT_PER_MIN` etc. | 120 / batch 6 / report 60 / metrics 30 / metrics refresh 6 / rings 60 | Per-client-IP budgets for the expensive endpoints (`PREDICT`, `BATCH`, `REPORT`, `METRICS`, `METRICS_REFRESH`, `RINGS`). `429` + `Retry-After` when exceeded. |

Other protections: request bodies are capped before parsing (16 KB JSON, 4 KB for the
report, ~1 MB for a batch upload; `413`), string/number bounds from Phase 1, bounded
query parameters (`limit`, `min_customers`), one batch at a time, and a lock so
`/metrics` is never computed by several requests at once.

### PDF report security

`POST /report/pdf` takes **only** `{"transaction_id": "TXN_..."}` (anything else is
rejected with `422`). The report is built from the server's own record of that
prediction (kept in memory for the most recent 1,000 predictions; a restarted server
returns `404`, so scan again), and the download filename is derived from the validated id.
Every value placed in the PDF is also XML-escaped. This matters because ReportLab's
`<img src="...">` markup embeds **any image file the server can read**; ReportLab's own
`trustedSchemes`/`trustedHosts` settings were tested and do not prevent that.

### Concurrency and chronological consistency

Each customer has its own lock (same-customer requests stay sequential, so no update is
lost); a batch holds only the locks of the customers it contains; and a short-lived
inference lock serialises just the Keras/SHAP calls, so an unrelated `/predict` waits for
at most one transaction's inference rather than a whole batch. A transaction is scored
against the history that precedes its own timestamp, and the "first time this
device/location was seen" flags of the entire history are re-derived chronologically on
every insert, so a back-dated transaction cannot leave later rows with stale flags.
Two rows with exactly identical timestamps are an untested edge case (their relative order is not guaranteed).
