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

# ---------------------------------------------------------------------------
# Request-validation bounds (applied at the API boundary, BEFORE any feature
# engineering / history update / model inference).
# The synthetic dataset's largest amount is ~Rs. 31,700 and its largest failed-
# login count is 15; these bounds are deliberately far above that so valid
# demo traffic is never rejected, while non-finite / absurd values are.
# ---------------------------------------------------------------------------
MAX_TRANSACTION_AMOUNT = 1_000_000.0   # Rs. 10 lakh per transaction
MAX_FAILED_LOGINS = 100
MAX_ID_LENGTH = 64                     # customer_id / device_id
MAX_TEXT_LENGTH = 64                   # location / merchant_category
MAX_FUTURE_TIMESTAMP_DAYS = 7          # supplied timestamps may not be further ahead than this
MIN_TIMESTAMP_YEAR = 2000

# Batch scoring runs ~0.6 s per row on CPU (2 Keras predicts + SHAP), so cap the
# request to keep it well inside a demo-friendly time budget.
BATCH_MAX_ROWS = 50
BATCH_MAX_BYTES = 1_000_000

# ---------------------------------------------------------------------------
# Cold start (Phase 2). The LSTM was trained ONLY on full 10-transaction
# windows of real history, and the behavioral-similarity baseline needs enough
# prior transactions for a meaningful mean/std. With fewer prior transactions
# than this, the LSTM risk score and similarity are reported as "not available"
# instead of being computed from a padded/fabricated window.
# ---------------------------------------------------------------------------
ESTABLISHED_HISTORY_MIN = SEQUENCE_LENGTH
RECENT_RESULTS_MAX = 1000              # server-side memory of recent predictions (used by /report/pdf)

# ---------------------------------------------------------------------------
# API protection (Phase 2) -- demo-level, in-process. Everything is overridable
# with real environment variables (this app does NOT read .env files itself).
# ---------------------------------------------------------------------------
def _env_list(name: str, default: list) -> list:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# Browser origins allowed to call the API cross-origin. The Vite dev server (and
# its /api proxy) runs on 5173. Set CORS_ORIGINS="https://your-frontend.example"
# for a deployed frontend; a lone "*" is honoured but never combined with credentials.
CORS_ORIGINS = _env_list("CORS_ORIGINS", ["http://localhost:5173", "http://127.0.0.1:5173"])

# Request-body ceilings (bytes), enforced before any parsing.
MAX_JSON_BODY_BYTES = 16_384
MAX_REPORT_BODY_BYTES = 4_096
MAX_BATCH_BODY_BYTES = BATCH_MAX_BYTES + 65_536   # file + multipart framing

# Per-client-IP request budgets: (max requests, window in seconds). Only the
# expensive endpoints are limited. RATE_LIMIT_ENABLED=0 switches them all off.
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1") not in ("0", "false", "False")
RATE_LIMITS = {
    "predict": (_env_int("RATE_LIMIT_PREDICT_PER_MIN", 120), 60),
    "batch": (_env_int("RATE_LIMIT_BATCH_PER_MIN", 6), 60),
    "report": (_env_int("RATE_LIMIT_REPORT_PER_MIN", 60), 60),
    "metrics": (_env_int("RATE_LIMIT_METRICS_PER_MIN", 30), 60),
    "metrics_refresh": (_env_int("RATE_LIMIT_METRICS_REFRESH_PER_MIN", 6), 60),
    "rings": (_env_int("RATE_LIMIT_RINGS_PER_MIN", 60), 60),
}
