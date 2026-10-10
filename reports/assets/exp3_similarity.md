| Score | ROC-AUC [95 % CI] | PR-AUC [95 % CI] |
|---|---|---|
| similarity deviation (served formula) | 0.866 [0.852, 0.878] | 0.069 [0.059, 0.081] |
| variant: std floored at 0.5 | 0.859 [0.846, 0.872] | 0.044 [0.038, 0.050] |
| default classifier score | 0.941 [0.932, 0.950] | 0.507 [0.466, 0.547] |
| classifier + similarity | 0.941 [0.932, 0.950] | 0.508 [0.467, 0.549] |

| Fraud type (against all legitimate rows) | Fraud rows | ROC-AUC [95 % CI] |
|---|---|---|
| account takeover | 647 | 0.895 [0.877, 0.913] |
| card testing cashout | 714 | 0.948 [0.937, 0.958] |
| device takeover | 159 | 0.929 [0.904, 0.950] |
| high value single | 105 | 0.989 [0.984, 0.993] |
| normal looking | 165 | 0.618 [0.569, 0.670] |
| ring | 262 | 0.972 [0.960, 0.982] |
| stolen card online | 858 | 0.763 [0.737, 0.790] |
