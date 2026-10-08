# Final review — architecture

## 1. Scoring path (one transaction)

```
Transaction
(customer, amount, merchant category, device, location, failed logins, time)
        │
        ▼
Feature Engineering                9 behavioral features, each computed only
        │                          from this customer's EARLIER transactions
        ▼
Customer Behavioral History        the customer's past transactions, kept in
        │                          time order (seed data + every scan so far)
        ▼
LSTM temporal risk model           reads the 10 transactions BEFORE this one
        │
        ▼
Temporal risk signal               Risk Score, 0–100
        │
        ▼
DNN Fraud Classifier               9 features of this transaction + Risk Score
        │
        ▼
Fraud Score                        0–100, a model score, not a probability
        │
        ├──────────────► SHAP Explanation      which features pushed the score up
        │
        ▼
Risk / Alert                       Low < 25 ≤ Medium < 50 ≤ High < 80 ≤ Critical
        │
        ▼
Investigation modules              Behavioral similarity, customer profile,
                                   score timeline, PDF report, batch scoring
```

### The stages

| Stage | What it does | Where |
|---|---|---|
| Feature engineering | Turns the raw transaction into 9 numbers that compare it with the customer's own past | `backend/app/features/feature_engineering.py` |
| Customer history | In-memory history per customer, loaded from the seed data and rebuilt from the database at start-up; every scored transaction is appended | `backend/app/inference_pipeline.py` |
| LSTM temporal risk model | Two LSTM layers (64 and 32 units) over a window of 10 earlier transactions × 9 features; output × 100 = Risk Score | `backend/app/models/lstm_model.py` |
| DNN fraud classifier | Dense layers 64 → 32 → 16 → 1 on 10 inputs (9 features + Risk Score); output × 100, capped at 99.9 = Fraud Score | `backend/app/models/dnn_model.py` |
| SHAP explanation | Feature attributions on the DNN; the top contributions towards a higher score are shown as reasons | `backend/app/models/shap_explainer.py` |
| Alert level | Fixed bands on the Fraud Score: 25 / 50 / 80 | `backend/app/models/dnn_model.py` |
| Behavioral similarity | How far the transaction's features are from the customer's own averages, in standard deviations, mapped to 0–100% | `backend/app/models/similarity.py` |

### The 9 features

| Feature | Meaning |
|---|---|
| `amount_zscore` | how many standard deviations the amount is from the customer's normal |
| `amount_pct_of_avg` | the amount as a percentage of the customer's average |
| `hour_is_unusual` | 1 if under 5% of the customer's past transactions were in this hour |
| `is_new_device` | 1 if the customer never used this device before |
| `is_new_location` | 1 if the customer never transacted from this location before |
| `is_foreign_location` | 1 if the location is outside the list of home cities |
| `category_is_unusual` | 1 if the merchant category is outside the customer's usual ones |
| `failed_logins_24h` | failed login attempts in the last 24 hours |
| `txn_velocity_1h` | the customer's transactions in the last hour |

### Why two stages

The LSTM models the customer's recent transaction sequence: it captures
temporal behavioral patterns and produces a temporal risk signal (the Risk
Score). The DNN looks at **this** transaction: is it abnormal for this customer
right now? The LSTM's output is one of the DNN's inputs, so the final score
uses both the recent sequence and the single event.

We do not claim that the LSTM predicts fraud before it happens. On v1 it
flagged none of the first-fraud transactions in the test period, and on v2
the LSTM + DNN design detects the first fraud of an episode *less* often than
a DNN without it (0.605 against 0.690 on the final hold-out).

### Cold start

Every model was trained only on transactions that have at least 10 earlier
ones. For a customer with fewer, the LSTM is not run and the DNN receives the
training-average Risk Score. Accuracy for such customers is weaker; see the
new-customer limitation in `final-review-results.md`.

## 2. Fraud Ring path

```
Customer accounts                  every customer's transaction history
        │
        ▼
Shared devices                     a device ID used by two or more customers
        │
        ▼
Relationship graph                 nodes: customers and devices
        │                          edges: "customer used the device"
        ▼
Suspicious clusters                connected groups of customers and devices
        │
        ▼
Investigation                      customers linked, shared devices, linked
                                   transactions, severity by cluster size
```

A legitimate customer's devices belong to that customer alone, so one device
appearing under several accounts may indicate coordinated activity and is worth investigating; it is not proof of fraud. The backend
returns each shared device with its customers and transaction count
(`GET /fraud-rings`). The frontend joins rings that share a customer into one
cluster and draws the graph. This part is a **rule over device IDs, not a
machine-learning model**. The severity label on a cluster is a reading aid
derived from its size (2 customers: Medium; 3 customers or 2 devices: High;
more: Critical).

## 3. System components

```
React + Vite frontend (port 5173)
  Live Scan │ Batch Scoring │ Fraud Rings │ Model Performance
        │  HTTP (/api → port 8000)
        ▼
FastAPI backend (port 8000)
  /predict  /predict/batch  /fraud-rings  /customers  /customer/{id}/profile
  /customer/{id}/history  /report/pdf  /model-info  /metrics  /health
        │
        ├── Inference pipeline: features → LSTM → DNN → SHAP → similarity
        ├── Model set (chosen by MODEL_SET at start-up; default "production")
        ├── SQLite database: every scored transaction, with the model set that scored it
        └── PDF report generator
```

| Layer | Technology |
|---|---|
| Models | TensorFlow / Keras (LSTM and DNN), SHAP |
| Backend | Python, FastAPI, SQLAlchemy, SQLite, ReportLab (PDF) |
| Frontend | React, Vite, Tailwind CSS, Recharts |

## 4. Model sets

| Model set | What it is | Status |
|---|---|---|
| `production` | LSTM + DNN trained on dataset v1; files in `backend/models/saved/` | **Loaded by default; the model in the demo** |
| `v2_dnn_lstm_seed14` | The same LSTM + DNN architecture retrained on dataset v2 with training seed 14; selected by the pre-registered protocol | Evaluation only, **not deployed** |
| `v2_dnn_lstm`, `v2_dnn_only` | Earlier v2 candidates (seed 42) | Evaluation only |

The model set is read once at start-up from the `MODEL_SET` environment
variable. Unset means `production`. An unknown name stops the server instead of
falling back. The seed-14 files are checked against recorded SHA-256 values
before loading.
