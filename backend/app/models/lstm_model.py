"""
LSTM Behavioral Risk Predictor
================================
The LSTM processes a customer's previous N (10) transactions as a sequence
to capture temporal behavioral patterns, and produces a temporal risk
signal, the Risk Score (0-100), for the downstream DNN fraud classifier.
Training target: whether the NEXT transaction after the window is
fraudulent; the output x100 is reported as the Risk Score. It is a model
score, not a calibrated probability.

We do not claim that it predicts fraud before it happens. The original
research hypothesis was that the score would rise before a customer's
first fraudulent transaction.
Measured result (docs/EVALUATION.md, time-based test split of the
synthetic data): the hypothesis is not supported. The score flags 0 of 16
first-fraud transactions, rises only after an episode has started (median
1.5 fraud transactions later), and the DNN performs the same with or
without it.

We train it as a binary classifier on "is the NEXT transaction after this
window fraudulent" and scale the predicted probability to 0-100 as the
Risk Score. Training uses class weights, so the probability is not
calibrated.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import (
    LSTM_X_PATH, LSTM_Y_PATH, LSTM_MODEL_PATH,
    LSTM_FEATURE_MEAN_PATH, LSTM_FEATURE_STD_PATH,
)


def build_lstm_model(seq_len: int, n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(seq_len, n_features), name="behavior_sequence")
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="risk_probability")(x)

    model = keras.Model(inputs, outputs, name="lstm_risk_predictor")
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


def risk_probability_to_score(prob: float) -> float:
    """Maps model output probability (0-1) to a Risk Score (0-100)."""
    return round(float(prob) * 100, 2)


if __name__ == "__main__":
    # Run this with:  cd backend && python -m app.models.lstm_model
    import os
    from sklearn.model_selection import train_test_split

    X = np.load(LSTM_X_PATH)
    y = np.load(LSTM_Y_PATH)
    print(f"Loaded sequences: X={X.shape}, y={y.shape}, fraud_rate={y.mean()*100:.2f}%")

    # normalize features (fit scaler on train split only, to avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_samples, seq_len, n_features = X_train.shape
    flat_train = X_train.reshape(-1, n_features)
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0)
    std[std == 0] = 1.0

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    np.save(LSTM_FEATURE_MEAN_PATH, mean)
    np.save(LSTM_FEATURE_STD_PATH, std)

    model = build_lstm_model(seq_len, n_features)
    model.summary()

    # class weights to handle heavy imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {0: 1.0, 1: max(n_neg / max(n_pos, 1), 1.0)}
    print(f"Class weights: {class_weight}")

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train_norm, y_train,
        validation_split=0.15,
        epochs=30,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    results = model.evaluate(X_test_norm, y_test, verbose=0)
    print("\nTest set results:")
    for name, val in zip(model.metrics_names, results):
        print(f"  {name}: {val:.4f}")

    model.save(LSTM_MODEL_PATH)
    print(f"\nSaved model to {LSTM_MODEL_PATH}")
