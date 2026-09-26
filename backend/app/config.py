"""
Central path configuration. All paths are resolved relative to this file's
location, so scripts work correctly no matter what directory they're run
from (as long as the folder structure is kept intact).
"""

import os
from pathlib import Path

# backend/app/config.py -> backend/app -> backend -> PROJECT_ROOT
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_SAVED_DIR = BACKEND_DIR / "models" / "saved"

MODELS_SAVED_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# raw + derived data files
RAW_TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
FEATURES_CSV = DATA_DIR / "transactions_with_features.csv"
LSTM_X_PATH = DATA_DIR / "lstm_X.npy"
LSTM_Y_PATH = DATA_DIR / "lstm_y.npy"
LSTM_META_CSV = DATA_DIR / "lstm_meta.csv"

# model artifacts
LSTM_MODEL_PATH = MODELS_SAVED_DIR / "lstm_risk_model.keras"
LSTM_FEATURE_MEAN_PATH = MODELS_SAVED_DIR / "lstm_feature_mean.npy"
LSTM_FEATURE_STD_PATH = MODELS_SAVED_DIR / "lstm_feature_std.npy"

DNN_MODEL_PATH = MODELS_SAVED_DIR / "dnn_fraud_model.keras"
DNN_FEATURE_MEAN_PATH = MODELS_SAVED_DIR / "dnn_feature_mean.npy"
DNN_FEATURE_STD_PATH = MODELS_SAVED_DIR / "dnn_feature_std.npy"
SHAP_BACKGROUND_PATH = MODELS_SAVED_DIR / "shap_background.npy"

SEQUENCE_LENGTH = 10

# corrected (time-based, leakage-safe) evaluation -- see app/evaluation/
EVALUATION_DIR = BACKEND_DIR / "models" / "evaluation"
EVAL_REPORT_PATH = EVALUATION_DIR / "evaluation_report.json"
EVAL_TIME_SPLIT_PATH = EVALUATION_DIR / "split_time.json"
EVAL_CUSTOMER_SPLIT_PATH = EVALUATION_DIR / "split_customer.json"


# ---- CORS ---------------------------------------------------------------------
# Browser origins allowed to call the API directly, as a comma-separated list
# in the CORS_ALLOW_ORIGINS environment variable, e.g.
#   CORS_ALLOW_ORIGINS=https://fraud-dashboard.example.com,http://localhost:5173
# Unset or blank -> the local Vite dev (5173) and preview (4173) servers.
# "*" allows any origin (credentials are then disabled; see main.py).
# In local development the dashboard calls the API through Vite's /api proxy,
# which is same-origin and doesn't depend on this setting at all.
DEFAULT_CORS_ALLOW_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def cors_allow_origins(raw=None) -> list[str]:
    """Parses CORS_ALLOW_ORIGINS (or `raw`, if given) into a list of origins.
    Whitespace and trailing slashes are stripped (a browser's Origin header
    never has a trailing slash, so "https://x.com/" would otherwise never match)."""
    if raw is None:
        raw = os.getenv("CORS_ALLOW_ORIGINS", "")
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return origins or list(DEFAULT_CORS_ALLOW_ORIGINS)
