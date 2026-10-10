"""
DNN Real-Time Fraud Detector
==============================
Takes CURRENT transaction's behavioral features + the LSTM-derived Risk
Score as an extra input, and outputs a Fraud Probability (0-100%) for this
single transaction. This is the real-time detection half of the system.
The LSTM Risk Score is its historical-risk input; on the current synthetic
data the DNN performs the same without it (docs/EVALUATION.md).

Input feature vector = FEATURE_COLUMNS (from feature_engineering.py) + risk_score
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import (
    FEATURES_CSV, LSTM_META_CSV, LSTM_X_PATH, LSTM_MODEL_PATH,
    LSTM_FEATURE_MEAN_PATH, LSTM_FEATURE_STD_PATH, DNN_MODEL_PATH,
    DNN_FEATURE_MEAN_PATH, DNN_FEATURE_STD_PATH, SHAP_BACKGROUND_PATH,
)

# same 9 behavioral features as the LSTM, plus the risk score from the LSTM
DNN_INPUT_COLUMNS = [
    "amount_zscore", "hour_is_unusual", "is_new_device", "is_new_location",
    "is_foreign_location", "failed_logins_24h", "category_is_unusual",
    "txn_velocity_1h", "amount_pct_of_avg", "risk_score",
]


def build_dnn_model(n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(n_features,), name="transaction_features")
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="fraud_probability")(x)

    model = keras.Model(inputs, outputs, name="dnn_fraud_detector")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def alert_level_from_probability(prob: float) -> str:
    pct = prob * 100
    if pct < 25:
        return "Low Risk"
    elif pct < 50:
        return "Medium Risk"
    elif pct < 80:
        return "High Risk"
    else:
        return "Critical Risk"


if __name__ == "__main__":
    # Run this with:  cd backend && python -m app.models.dnn_model
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from .lstm_model import risk_probability_to_score

    feat_df = pd.read_csv(FEATURES_CSV)
    lstm_meta = pd.read_csv(LSTM_META_CSV)

    # get LSTM risk scores for every sequence we built, then join onto the
    # matching transaction row so the DNN can use "risk score at the time"
    # as an input feature (this is the LSTM -> DNN handoff from the objectives)
    lstm_model = keras.models.load_model(LSTM_MODEL_PATH)
    X_seq = np.load(LSTM_X_PATH)
    mean = np.load(LSTM_FEATURE_MEAN_PATH)
    std = np.load(LSTM_FEATURE_STD_PATH)
    X_seq_norm = (X_seq - mean) / std

    print("Scoring all sequences with trained LSTM to derive risk_score feature...")
    risk_probs = lstm_model.predict(X_seq_norm, batch_size=512, verbose=1).flatten()
    lstm_meta["risk_score"] = [risk_probability_to_score(p) for p in risk_probs]

    merged = feat_df.merge(
        lstm_meta[["transaction_id", "risk_score"]], on="transaction_id", how="inner"
    )
    print(f"Merged {len(merged):,} rows with risk_score attached")

    # For transactions that didn't get a risk_score (first SEQUENCE_LENGTH per
    # customer, before enough history existed), fill with the median - these
    # are cold-start rows and get dropped from training but this keeps things
    # robust if used elsewhere.
    X = merged[DNN_INPUT_COLUMNS].values.astype(np.float32)
    y = merged["is_fraud"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dnn_mean = X_train.mean(axis=0)
    dnn_std = X_train.std(axis=0)
    dnn_std[dnn_std == 0] = 1.0
    X_train_norm = (X_train - dnn_mean) / dnn_std
    X_test_norm = (X_test - dnn_mean) / dnn_std

    np.save(DNN_FEATURE_MEAN_PATH, dnn_mean)
    np.save(DNN_FEATURE_STD_PATH, dnn_std)

    model = build_dnn_model(X_train.shape[1])
    model.summary()

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: 1.0, 1: max(n_neg / max(n_pos, 1), 1.0)}
    print(f"Class weights: {class_weight}")

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train_norm, y_train,
        validation_split=0.15,
        epochs=40,
        batch_size=128,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    results = model.evaluate(X_test_norm, y_test, verbose=0)
    print("\nTest set results:")
    for name, val in zip(model.metrics_names, results):
        print(f"  {name}: {val:.4f}")

    model.save(DNN_MODEL_PATH)
    print(f"\nSaved model to {DNN_MODEL_PATH}")

    # save a small background sample for SHAP (needs a reference dataset)
    background_sample = X_train_norm[np.random.choice(len(X_train_norm), size=200, replace=False)]
    np.save(SHAP_BACKGROUND_PATH, background_sample)
