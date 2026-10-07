"""
Training / inference feature-skew check
=========================================
The shipped models were trained on data/transactions_with_features.csv, which was
generated BEFORE two preprocessing changes now in feature_engineering.py:

  * amount_zscore: the std-dev is floored at max(std, 10% of the average, 1.0) and the
    z-score is clipped to [-10, 10] (the shipped CSV still holds z-scores up to +/-1,585)
  * amount_pct_of_avg: clipped to [0, 1000] (the shipped CSV holds values up to 8,440)

Live inference recomputes features with the CURRENT code, so a transaction's
amount_zscore at inference can differ from what the same transaction would have
had in the training file. This script measures whether that matters: it scores the
SHIPPED models on (a) the shipped feature file and (b) the same raw transactions with
features recomputed by the current code, on the same held-out split used by
evaluate.py.

Run from backend/ (takes ~2 minutes):
    python -m app.models.skew_check

The shipped models are deliberately NOT retrained or changed. See README
("Known limitations") for the recorded result and the reasoning.
"""

import pandas as pd

from ..config import FEATURES_CSV
from ..features.feature_engineering import build_point_features, build_sequences, FEATURE_COLUMNS
from .evaluate import _evaluate_dnn, _evaluate_lstm

RAW_COLUMNS = [
    "customer_id", "transaction_id", "timestamp", "amount", "merchant_category", "device_id",
    "location", "failed_logins_24h", "is_new_device", "is_new_location", "is_fraud",
]


def _evaluate(feat_df: pd.DataFrame) -> dict:
    X_seq, y_seq, meta = build_sequences(feat_df)
    return {"lstm": _evaluate_lstm(X_seq, y_seq), "dnn": _evaluate_dnn(feat_df, X_seq, meta)}


def main():
    shipped = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    print(f"Recomputing features for {len(shipped):,} rows with the current feature code...")
    current = build_point_features(shipped[RAW_COLUMNS])

    a = shipped.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    b = current.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    print("\nRows whose feature value differs between the shipped file and the current code:")
    for col in FEATURE_COLUMNS:
        n = int(((a[col].astype(float) - b[col].astype(float)).abs() > 1e-6).sum())
        if n:
            print(f"  {col:20s} {n:6d} of {len(a):,} rows")
    print(f"Shipped file extremes: amount_zscore {a['amount_zscore'].min():.0f}..{a['amount_zscore'].max():.0f}, "
          f"amount_pct_of_avg max {a['amount_pct_of_avg'].max():.0f}")

    for tag, df in (("shipped CSV features (what the models were trained on)", shipped),
                    ("current-code features (what live inference computes)", current)):
        r = _evaluate(df)
        print(f"\n[{tag}]")
        for name in ("lstm", "dnn"):
            m = r[name]
            cm = m["confusion_matrix"]
            print(f"  {name.upper():4s} precision {m['precision']:.4f}  recall {m['recall']:.4f}  ROC-AUC {m['auc_roc']:.4f}  "
                  f"PR-AUC {m['pr_auc']:.4f}  FP {cm['false_positive']}  FN {cm['false_negative']}")


if __name__ == "__main__":
    main()
