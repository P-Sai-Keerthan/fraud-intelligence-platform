"""
Model performance for GET /metrics and GET /metrics/report.

The numbers come from the corrected, leakage-safe evaluation in
app/evaluation/ (time-based split, out-of-fold LSTM -> DNN stacking,
thresholds chosen on validation). Training that evaluation takes ~11 min on a 2-core CPU,
so it runs offline:

    cd backend && python -m app.evaluation.run

and writes models/evaluation/evaluation_report.json, which this module
serves. The report is read once and cached; refresh re-reads the file
(it does not retrain).

Until Step 4B this module recomputed metrics for the production models on a
random stratified 80/20 split at threshold 0.5. That method is kept in the
report only as "legacy_random_split", for comparison.
"""

import json

from ..config import EVAL_REPORT_PATH

MODEL_KEYS = ("lstm_risk_predictor", "dnn_fraud_classifier")

_cache = None


class EvaluationReportMissing(RuntimeError):
    pass


def load_report(force_refresh: bool = False) -> dict:
    global _cache
    if _cache is None or force_refresh:
        if not EVAL_REPORT_PATH.exists():
            raise EvaluationReportMissing(
                f"{EVAL_REPORT_PATH.name} not found - run `python -m app.evaluation.run` from backend/ to create it"
            )
        _cache = json.loads(EVAL_REPORT_PATH.read_text())
    return _cache


def evaluate_all(force_refresh: bool = False) -> dict:
    """The two production-model entries of the corrected (primary, time-based)
    evaluation, in the same shape /metrics has always returned."""
    report = load_report(force_refresh)
    return {key: report["primary"][key] for key in MODEL_KEYS}


if __name__ == "__main__":
    # Run this with:  cd backend && python -m app.models.evaluate
    print(json.dumps(evaluate_all(), indent=2))
