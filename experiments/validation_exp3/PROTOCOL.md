# Experiment 3 — similarity, calibration and ablation checks (protocol EXP3 v1)

**Status when this file was committed: no data for this experiment had been generated and no EXP3 result existed.** The commit that adds this
file precedes the commit that adds any code, data manifest or result of the experiment; its SHA-256 is recorded in every result file.

Scope: *synthetic data only.* Nothing here validates the system on real banking data. The saved models, saved scores, the single-use hold-outs
(seeds 401–405, 411–415, 501–505, 511–515) and all earlier evaluation artifacts are **read only** and are never modified or re-scored.
Experiment 2 (`experiments/sequence_early_detection/`) is untouched; its reserved seeds (7101–7103, 7201–7203, 8101–8110, 8201–8205) are not used.

## 1. Data

| Role | Seeds | Generator | Customers per seed |
|---|---|---|---|
| Validation ("DEV3") | 9301, 9302 | `data/v2/generate.py`, generator 2.0.1, default configuration | 1,000 |
| Test ("TEST3") | 9303, 9304, 9305 | same | 1,000 |

* All five seeds are new: none appears in `SPENT_SEEDS`, in the Experiment 2 reservation, or in any file of the repository (checked by search before use).
* The models under test were trained on the seed-42 v2 dataset only, so every row of these datasets is out of sample for them.
* **Role discipline:** anything that is *fitted or chosen* in this experiment (calibrators, the similarity-combination model, alert thresholds, the re-trained
  baselines' operating points) is fitted on DEV3 only. TEST3 is used once for the numbers reported. No setting is changed after TEST3 has been looked at; if
  something must change, it is recorded as a deviation with the reason.
* Datasets are written outside the repository (they are large and deterministic); their SHA-256 and the generator command are recorded in `data_manifests/`.
  Determinism of the generator across environments was verified in the audit (seed 42 regenerated, all five hashes matched).

Expected size per seed: ≈200 k transactions and ≈940 fraud rows (planning figures from a 2,000-customer scratch run; actual counts are reported).

## 2. Models and populations

* **Default:** `v2_lstm_rf_seed14` (LSTM → random forest), the model set served by default.
* **Comparators already saved:** `v2_dnn_lstm_seed14` (LSTM → DNN), `v2_dnn_only` (DNN on the nine features, no LSTM), `production` (v1 LSTM → DNN; trained on a different dataset, shown for context only).
* **Re-trained baselines (new, trained here on the seed-42 v2 *training period* only, i.e. the rows with ≥ 10 earlier transactions before 2026-05-03, exactly the rows the default model was trained on):**
  random forest on the nine features **without** the LSTM risk input (same hyper-parameters as the default: 200 trees, depth 12, min leaf 5, seed 14); logistic regression on the nine features (standardised, default settings); the project's amount-and-hour rule.
* **Similarity:** the current formula, unchanged (`app/models/similarity.py::compute_similarity`), evaluated per transaction against the mean and standard deviation of the customer's earlier feature vectors. Variants are separate and labelled:
  * `S0` current formula (zero or near-zero std replaced by 1 only if ≤ 1e-6);
  * `S1` exploratory variant: per-feature std floored at 0.5. It is **not** a proposed change; it isolates the effect of near-constant features.
* **Populations:** all scoring uses each row's own past only. Two populations are reported: **P10** = rows with ≥ 10 earlier transactions (what the API now scores for similarity; the LSTM population) and **ALL** rows. Comparisons between models use identical rows.

## 3. Research questions, analyses, success criteria

### EXP3-SIM — is the Behavioral Similarity Score informative?
Deviation score `d = 100 − similarity` (higher = less like the customer's norm). Direction is **not assumed**; the analysis reports it.
1. Distributions of `d` for legitimate vs fraud rows (median, IQR) overall, by fraud type, and by history length bin (0, 1–9, 10–29, 30–99, ≥100 earlier transactions).
2. Discrimination: ROC-AUC and PR-AUC of `d` as a fraud score, with 95 % customer-cluster bootstrap intervals (2,000 resamples of customers within each seed, seed 9101); effect size as Cliff's δ = 2·AUC − 1. AUC < 0.5 means fraud looks *more* similar than legitimate rows.
3. Incremental value: a logistic regression on `[logit(clipped classifier score), d]` fitted on DEV3, compared on TEST3 with the classifier score alone (paired bootstrap of PR-AUC and ROC-AUC differences). Also report the within-fraud correlation of `d` with the classifier score.
4. History length: AUC of `d` in each history bin (P10 and below).
5. Zero-variance and near-constant features: share of rows where ≥ 1 of the nine prior-history standard deviations is ≤ 1e-6 or below 0.1; AUC of S0 vs S1.
6. Circularity check, stated up front: fraud in this generator deviates from the generator's customer baselines by construction (new device, foreign city, large amount), and the similarity uses the same features as the classifier. A positive AUC therefore shows only that the score reflects the generator's fraud rules; it is not evidence about real fraud.
**Interpretation rule (pre-specified):** the score "has discriminatory value on this data" if the AUC interval on P10 excludes 0.5; it "adds information beyond the classifier" only if the paired PR-AUC difference interval excludes 0. Otherwise it is reported as not demonstrated.

### EXP3-CAL — is the Fraud Score a calibrated probability?
Score used: the default model's output in [0, 1] (the serving value is this × 100, capped at 99.9; the cap is applied).
1. Measure the **uncalibrated** score as if it were a probability: Brier score, log loss, ECE (15 equal-width and 15 equal-mass bins), reliability table, calibration slope and intercept, on TEST3.
2. Fit **sigmoid (Platt)** and **isotonic** calibrators on DEV3 only; evaluate on TEST3 with the same metrics; reference: predicting the DEV3 prevalence.
3. Stability: results per test seed and on P10.
4. Ranking metrics (PR-AUC, ROC-AUC) before and after (isotonic may create ties).
**Rule:** a calibrator is called useful only if its TEST3 Brier score and log loss are lower than the uncalibrated score's with 95 % intervals excluding 0 *and* the effect holds in each of the three test seeds. Nothing is deployed; the serving score is not renamed or reinterpreted.

### EXP3-ABL — baselines and ablations on identical rows
Models: default, LSTM → DNN, DNN-only, production, RF without LSTM, logistic regression, amount-and-hour rule, similarity alone (`d`), default + similarity (EXP3-SIM combination).
Alert threshold per model: the lowest score cut-off whose **DEV3 false-positive rate is ≤ 1 %** (tie-safe, `multiseed.tie_safe_cutoff`), applied unchanged to TEST3.
Reported on TEST3 (P10 and ALL): class counts, PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix, legitimate alerts per 1,000, first-fraud recall, with 95 % customer-cluster bootstrap intervals; paired differences for (default − RF without LSTM) and (default − LSTM→DNN). Accuracy is not used as evidence.
**Interpretation:** an ablation effect is reported as detected only when its paired interval excludes 0.

## 4. Reproducibility record

Each result file carries: experiment id, this protocol's SHA-256, seeds, dataset SHA-256 (from the generator manifests), the model-artifact SHA-256 values (from the repository's pinned manifests), software versions (Python, numpy, pandas, scikit-learn, TensorFlow), the exact commands, and the bootstrap seed. Raw per-row scores are written next to the results (compressed) so every table can be recomputed without the models.

Commands (run from the repository root, in the backend virtual environment):

```
python experiments/validation_exp3/run_experiment.py generate --out <data_dir>      # five datasets, manifests copied to data_manifests/
python experiments/validation_exp3/run_experiment.py score    --data <data_dir>     # per-row scores and similarity -> results/scores/
python experiments/validation_exp3/run_experiment.py analyze                         # all tables -> results/*.json, figures
```

## 5. Known limits stated before the results

* Synthetic data from the generator the models were developed on: no statement about real banking data, and generalisation to another generator is not tested.
* Test seeds share the generator's assumptions; the three test datasets are independent draws but not independent *worlds*.
* Calibration fitted on 2 datasets and tested on 3 assumes the same base rate (about 0.46 % fraud) and customer mix; a different rate would break it.
* The re-trained baselines use the repository's recipe and are not tuned; they are comparators, not competitors optimised for the task.

## 6. Amendment A1 (made before any EXP3 data was generated)

Clarifications that remove ambiguity; no result existed when they were written.

1. **Logistic regression and rule baselines use the repository's existing recipe** (`app/evaluation/baselines.py`): standardised nine features, class-balanced logistic regression (seed 42), fitted on the same training rows as the random-forest baseline; the rule scores `amount_pct_of_avg` when `hour_is_unusual` is set, else 0. (Section 2 said "default settings"; the existing recipe is used so the baseline is the one already defined in the project.)
2. **Pooled primary analysis.** Point estimates are reported per test seed and pooled over TEST3 (9303–9305). Bootstrap intervals are for the pooled TEST3 set: customer components (rings and households kept whole) are resampled with replacement within each seed, then pooled; 2,000 replicates, seed 9101. Per-seed results carry no interval.
3. **First-fraud recall** = share of fraud episodes whose first fraud transaction is alerted.
4. **Raw scores.** Per-row score files are written under `--out`; they are committed only if their total size is below 25 MB, otherwise their SHA-256 is recorded and they are regenerated by the `score` command.
5. **Similarity ties.** S0 is the served formula including its rounding (2 decimals). S1 uses the unrounded value.
6. **Calibrator inputs.** Calibration is applied to the default model's raw 0–1 score capped at 0.999 (as served); the sigmoid calibrator is fitted on the logit of the clipped score.
