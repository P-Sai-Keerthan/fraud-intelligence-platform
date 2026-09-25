# Explainable Fraud Intelligence Platform

An AI-powered banking fraud intelligence system that builds a **Behavioral
Fraud DNA** profile per customer, predicts risk *before* fraud happens
(LSTM), detects fraud in real time on individual transactions (DNN), and
explains every decision (SHAP).

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
│       ├── schemas.py                   # request/response models
│       ├── features/feature_engineering.py   # Behavioral Fraud DNA feature builder
│       ├── models/
│       │   ├── lstm_model.py            # LSTM risk predictor
│       │   ├── dnn_model.py              # DNN fraud classifier
│       │   ├── shap_explainer.py          # Explainable AI (SHAP) wrapper
│       │   └── similarity.py              # Behavioral Similarity Score
│       └── db/                          # SQLAlchemy models + session (SQLite by default)
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

- **Python 3.12 or newer** (tested on 3.12 and 3.13; the pinned `shap`/`numpy`/`pandas` versions don't install on 3.10/3.11) — check with `python3 --version`
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
[startup] Restored 0 previously scored transaction(s) from the database.
INFO:     Application startup complete.
```

Every transaction you score is saved to the database. When the server
restarts, those saved transactions are replayed into each customer's
behavioral history, so the models still know about them (for example, a
device first used before the restart is not treated as "new" after it).

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

### 3.5 Run the backend test suite

With the virtual environment activated, from inside `backend/`:

```bash
pip install -r requirements-dev.txt   # pytest, httpx, pypdf (test-only)
pytest                                # full suite, ~1 minute
pytest -m "not slow"                  # skip the slowest tests (metrics, restart, env-var checks)
```

The tests use a throwaway SQLite database in a temp folder, so they never
touch `fraud_platform.db`, and they don't need the API server running.

### 3.6 Configuration (environment variables)

All settings are optional; the defaults work for local development.
`backend/.env.example` lists them.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./fraud_platform.db` | Database for scored transactions (PostgreSQL also works). |
| `CORS_ALLOW_ORIGINS` | `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:4173`, `http://127.0.0.1:4173` | Comma-separated browser origins allowed to call the API directly. |

The dashboard's dev server reaches the API through Vite's `/api` proxy,
which is same-origin, so local development needs no CORS setup at all.
Set `CORS_ALLOW_ORIGINS` only when the frontend is served from another
origin and calls the backend directly (i.e. `VITE_API_BASE_URL` is set),
for example `CORS_ALLOW_ORIGINS=https://fraud-dashboard.example.com`.
`*` allows any origin, with credentials disabled.

Set variables in the shell before starting uvicorn, or put them in
`backend/.env` and start the server with `--env-file .env`:

```bash
cp .env.example .env        # then edit .env
uvicorn app.main:app --reload --port 8000 --env-file .env
```

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
You'll see the risk gauge, fraud probability, alert banner, SHAP
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
2. **LSTM Risk Predictor** (`lstm_model.py`): takes the customer's last 10
   transactions' behavioral features as a sequence, predicts the
   probability that the *next* transaction will be fraudulent. This score
   (0-100) represents risk building up *before* an attack.
3. **DNN Fraud Detector** (`dnn_model.py`): takes the current transaction's
   own behavioral features *plus* the LSTM risk score, and outputs a
   calibrated fraud probability for *this specific transaction*.
4. **SHAP Explainer** (`shap_explainer.py`): wraps the DNN with
   `shap.GradientExplainer` and maps the top contributing features to
   human-readable reasons ("New Device", "Foreign Location", etc.).
5. **Behavioral Similarity Score** (`similarity.py`): z-scores the current
   transaction against the customer's own historical mean/std per feature,
   and maps the average deviation to a 0-100 similarity percentage via
   exponential decay.

### Known limitations (be upfront about these in your paper — reviewers expect it)

- The dataset is **synthetic**. Real bank data is never public, so this is
  standard practice in fraud-detection research, but say so explicitly in
  your methodology section.
- Behavioral features are recomputed from a customer's full history on
  every request (`O(n)` per prediction). Fine at demo scale; a production
  system would maintain incrementally-updated rolling statistics instead.
- The Behavioral Similarity Score weights all 9 features equally. It can
  diverge from the DNN's fraud probability for customers with naturally
  tight variance in one dimension (e.g. very consistent spending amounts).
  This is a good "future work" paragraph for your paper: learned feature
  weighting for the similarity metric.
- Report **Precision, Recall, F1, and AUC-ROC** in your results — not just
  accuracy, since fraud is heavily imbalanced and accuracy alone looks
  artificially high.

See `docs/RESEARCH_NOTES.md` for a full IEEE/Springer paper structure
template and suggested related-work citations.

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
