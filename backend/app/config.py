"""
Central path configuration. All paths are resolved relative to this file's
location, so scripts work correctly no matter what directory they're run
from (as long as the folder structure is kept intact).
"""

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
