"""
Corrected, leakage-safe model evaluation.

    python -m app.evaluation.run      (from backend/)

trains evaluation copies of the LSTM and DNN with the corrected methodology
and writes models/evaluation/evaluation_report.json, which GET /metrics and
GET /metrics/report serve. The production models used by /predict are not
touched. See split.py (time/customer splits), windows.py (past-only LSTM
windows), stacking.py (out-of-fold LSTM -> DNN handoff), metrics.py
(validation-selected thresholds, episode metrics) and run.py (orchestration).
"""
