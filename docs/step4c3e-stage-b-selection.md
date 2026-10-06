# Step 4C-3E.6: Stage B artifact selection (result)

Run under the approved protocol `4C-3E.6 v1`
(`docs/step4c3e-model-selection-protocol.md`). The protocol and its code
were not changed after approval (file checksums identical before and after
the run).

**NO PROMOTION.** `MODEL_SET` remains `production`. No model was trained.
Production, seed-42 candidate and multi-seed model files are unchanged. The
final hold-out (seeds 401–405, 411–415) was **not generated and not
scored**. Nothing was deployed, committed or pushed.

## Result

| Role | Selected artifact | Directory |
|---|---|---|
| Primary candidate | `v2_dnn_lstm`, training seed **14** | `backend/models/candidates_multiseed/v2/seed_14/dnn_lstm/` |
| Required challenger | `v2_dnn_only`, training seed **14** | `backend/models/candidates_multiseed/v2/seed_14/dnn_only/` |

Both architectures passed Stage B. The record is
`backend/models/evaluation/v2_holdout/selection_record.json`. It is written
once; the code refuses to run the selection again.

| Artifact | Weights SHA-256 (first 12) | Policy B cut-off | Critical cut-off |
|---|---|---|---|
| `v2_dnn_lstm` seed 14 | LSTM `e4851c4c20e8`, DNN `32fc53979e06` | 0.7929 | 0.9431 |
| `v2_dnn_only` seed 14 | DNN `8b79fbf09a44` | 0.8425 | 0.9654 |

The cut-offs are the frozen validation-rule values. They were not
recalibrated.

## Development data

Seeds 301–305, default generator 2.0.1, 500 customers each, under
`data/v2_holdout/development/` (untracked). Primary population pooled:
489,898 transactions, 2,430 fraud transactions, 550 fraud episodes.

| Seed | Transactions | SHA-256 of the features file (first 12) |
|---|---|---|
| 301 | 102,596 | `59b481874e99` |
| 302 | 105,382 | `b72dae9cafcd` |
| 303 | 102,229 | `528368dd1eb6` |
| 304 | 101,499 | `5cc3ee9bcc72` |
| 305 | 103,192 | `1ccd00adbd28` |

## v2_dnn_lstm

| Seed | Recall | Legit alerts / 1,000 | Critical recall | First-fraud detection | Episodes with any alert | Eligible | Distance |
|---|---|---|---|---|---|---|---|
| 11 | 0.491 | 8.81 | 0.231 | 0.573 | 0.673 | yes | 3.858 |
| 12 | 0.550 | 10.65 | 0.205 | 0.545 | 0.695 | yes | 2.115 |
| 13 | 0.428 | 9.53 | 0.178 | 0.460 | 0.653 | yes | 7.392 |
| **14** | 0.556 | 8.93 | 0.214 | 0.571 | 0.705 | yes | **1.611** |
| 15 | 0.549 | 9.07 | 0.160 | 0.502 | 0.702 | yes | 2.824 |
| Median | 0.549 | 9.07 | 0.205 | 0.545 | 0.695 | | |

* 5 of 5 seeds eligible (3 needed). Seed 14 has the smallest distance.
* **Seed 14 also has the highest development recall**, by 0.006 over seeds 12
  and 15. Three seeds sit at 0.549–0.556, so the median is in that group and
  the most typical seed comes from it. Recall was not maximised: seed 14 is
  selected because it is close to the median on all five metrics (seed 12 is
  far on alert burden, seed 15 on Critical recall).
* Seed 14 was also the highest-recall `v2_dnn_lstm` run on the existing
  hold-out (0.573). The rule did not read that data. It still means the
  primary candidate is at the upper end of its architecture on recall, not
  in the middle. The final result should be read as one trained model, not
  as the architecture's average.

## v2_dnn_only

| Seed | Recall | Legit alerts / 1,000 | Critical recall | First-fraud detection | Episodes with any alert | Eligible | Distance |
|---|---|---|---|---|---|---|---|
| 11 | 0.515 | 10.32 | 0.152 | 0.722 | 0.771 | yes | 5.296 |
| 12 | 0.361 | 9.23 | 0.094 | 0.504 | 0.547 | **no** | 8.220 |
| 13 | 0.414 | 9.68 | 0.200 | 0.662 | 0.702 | yes | 0.765 |
| **14** | 0.433 | 9.75 | 0.189 | 0.665 | 0.698 | yes | **0.042** |
| 15 | 0.444 | 9.83 | 0.216 | 0.571 | 0.655 | yes | 2.567 |
| Median | 0.433 | 9.75 | 0.189 | 0.662 | 0.698 | | |

* 4 of 5 seeds eligible (3 needed). Seed 12 fails two screens: recall below
  0.40 and Critical recall more than 0.05 below the median.
* Seed 14 is the median run on four of the five metrics. The highest-recall
  seed is 11; it was not selected.

## What the development data shows (description only)

* The architecture gap is the same as on the earlier hold-out: median recall
  0.549 against 0.433; first-fraud detection 0.545 against 0.662.
* Alert burden at the frozen cut-offs: 8.8 to 10.7 per 1,000. Two of the ten
  models are above 10 (`v2_dnn_lstm` seed 12, `v2_dnn_only` seed 11). The
  selected artifacts are at 8.93 and 9.75.
* **The final test is unchanged:** upper bound of the interval ≤ 10
  legitimate alerts per 1,000. The `v2_dnn_only` artifact at 9.75 on
  development data leaves little room; it may fail that test on the final
  hold-out through dataset variation alone.

## Next step (not started)

Stage C: generate the final hold-out (401–405 and 411–415) and score the two
selected artifacts and `production` once, by the rule in §7 of the protocol.
It needs approval. Nothing in the protocol may change now.

## State

| Item | State |
|---|---|
| Promotion | **NO PROMOTION** |
| `MODEL_SET` | `production` |
| `backend/models/saved/`, `backend/models/candidates/v2/` | unchanged |
| `backend/models/candidates_multiseed/` | unchanged (70 files) |
| Selection protocol and `selection.py` | unchanged since approval |
| Final hold-out | not generated, not scored |
| Commit or push | none; HEAD `582fe60` |

**Files added:** `backend/models/evaluation/v2_holdout/selection_record.json`,
this document. **File changed:** `backend/tests/test_selection.py` (the test
that asserted "no selection record exists yet" now checks the record against
the rule and the artifacts on disk; 29 tests pass).

The development datasets are in the cloud workspace only. They are
reproducible from the seeds with the default generator; the record holds
their checksums.
