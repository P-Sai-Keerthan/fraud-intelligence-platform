*5 fresh datasets, 450,095 transactions, 2,693 fraud transactions, 632 fraud episodes, 549 first-fraud transactions.*

| Model | PR-AUC [95% CI] | ROC-AUC | Recall [95% CI] | Precision | F1 | Legit alerts / 1,000 | First-fraud recall [95% CI] | Episodes detected |
|---|---|---|---|---|---|---|---|---|
| LSTM + random forest (default) | 0.490 [0.449, 0.531] | 0.939 | 0.638 [0.603, 0.675] | 0.287 | 0.396 | 9.50 | 0.738 [0.696, 0.776] | 492 / 632 |
| LSTM + DNN (seed 14) | 0.365 [0.323, 0.408] | 0.928 | 0.533 [0.495, 0.572] | 0.266 | 0.355 | 8.78 | 0.599 [0.555, 0.642] | 440 / 632 |
| Previous production (v1 LSTM + DNN) | 0.163 [0.138, 0.188] | 0.705 | 0.269 [0.241, 0.297] | 0.133 | 0.178 | 10.49 | 0.399 [0.355, 0.443] | 366 / 632 |
