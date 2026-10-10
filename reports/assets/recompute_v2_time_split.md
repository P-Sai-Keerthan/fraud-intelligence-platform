*Test period: 21,142 windows, 61 fraud rows (from the committed `scores_time_split.csv.gz`).*

| Model | PR-AUC recomputed | PR-AUC in report | ROC-AUC recomputed | ROC-AUC in report |
|---|---|---|---|---|
| lstm_risk_predictor | 0.1345 | 0.1345 | 0.7407 | 0.7407 |
| dnn_fraud_classifier | 0.1045 | 0.1045 | 0.8993 | 0.8993 |
| amount_hour_rule | 0.0272 | 0.0272 | 0.7296 | 0.7296 |
| logistic_regression | 0.1375 | 0.1375 | 0.8905 | 0.8905 |
| dnn_without_risk_score | 0.1283 | 0.1283 | 0.8965 | 0.8965 |
