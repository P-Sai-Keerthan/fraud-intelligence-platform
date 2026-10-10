| Model | PR-AUC [95 % CI] | ROC-AUC | Recall | Precision | Legit. alerts per 1,000 legit. |
|---|---|---|---|---|---|
| default (LSTM -> random forest) | 0.507 [0.466, 0.547] | 0.941 | 0.642 | 0.251 | 9.48 |
| LSTM -> DNN | 0.373 [0.331, 0.413] | 0.932 | 0.558 | 0.220 | 9.80 |
| DNN only (no LSTM) | 0.209 [0.177, 0.242] | 0.893 | 0.405 | 0.173 | 9.57 |
| production v1 (LSTM -> DNN, different training data) | 0.172 [0.147, 0.198] | 0.709 | 0.279 | 0.125 | 9.71 |
| random forest, no LSTM input | 0.424 [0.383, 0.464] | 0.910 | 0.560 | 0.224 | 9.62 |
| logistic regression (balanced) | 0.289 [0.253, 0.325] | 0.891 | 0.411 | 0.173 | 9.72 |
| amount-and-hour rule | 0.032 [0.028, 0.038] | 0.666 | 0.201 | 0.089 | 10.23 |
| similarity deviation alone (S0) | 0.069 [0.059, 0.081] | 0.866 | 0.238 | 0.102 | 10.39 |
| default + similarity (logistic combination) | 0.508 [0.467, 0.549] | 0.941 | 0.638 | 0.248 | 9.55 |
