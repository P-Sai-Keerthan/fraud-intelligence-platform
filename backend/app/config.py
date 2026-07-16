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
