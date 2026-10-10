*5 fresh datasets, 487,368 transactions, 2,390 fraud transactions, 550 fraud episodes, 550 first-fraud transactions.*

| Model | PR-AUC [95% CI] | ROC-AUC | Recall [95% CI] | Precision | F1 | Legit alerts / 1,000 | First-fraud recall [95% CI] | Episodes detected |
|---|---|---|---|---|---|---|---|---|
| LSTM + random forest (default) | 0.464 [0.419, 0.509] | 0.939 | 0.645 [0.608, 0.683] | 0.236 | 0.346 | 10.23 | 0.738 [0.699, 0.776] | 439 / 550 |
| LSTM + DNN (seed 14) | 0.328 [0.286, 0.371] | 0.929 | 0.548 [0.505, 0.588] | 0.217 | 0.311 | 9.68 | 0.595 [0.550, 0.637] | 386 / 550 |
| Previous production (v1 LSTM + DNN) | 0.133 [0.111, 0.157] | 0.691 | 0.255 [0.228, 0.283] | 0.104 | 0.148 | 10.71 | 0.384 [0.340, 0.427] | 322 / 550 |
