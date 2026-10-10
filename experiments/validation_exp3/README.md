# Experiment 3 — Behavioral Similarity, calibration and ablation checks

Synthetic data only. Pre-registered in `PROTOCOL.md` (committed before any data existed); results in `RESULTS.md`.

| File | Content |
|---|---|
| `PROTOCOL.md` | questions, data, metrics, decision rules, amendment A1 |
| `run_experiment.py` | `generate`, `score`, `analyze` stages |
| `verify_results.py` | recomputes 51 headline numbers with scikit-learn from the per-row scores (all matched) |
| `make_summary.py` | writes `RESULTS.md` from `results/*.json` |
| `RESULTS.md` | all result tables (generated) |
| `results/exp3_{similarity,calibration,ablation}.json` | raw metrics, intervals, configuration and reproducibility metadata |
| `results/reproducibility_check.json`, `results/logs/` | the second full run compared with the first; command logs |
| `data_manifests/` | generator manifests (with dataset SHA-256) of the five datasets |
| `INDEPENDENT_VALIDATION_STATUS.md` | what independent evidence is missing and what a valid test needs |

Data: validation seeds 9301–9302, test seeds 9303–9305, 1,000 customers each, `data/v2/generate.py` (generator 2.0.1). Datasets and per-row scores are written outside the repository
(about 340 MB and 54 MB); they are regenerated bit for bit from the manifests' commands, and each results file records the SHA-256 of every dataset and of the decompressed score files.

Reproduce (backend virtual environment; the seed-42 v2 dataset must first be regenerated into a scratch directory with `python data/v2/generate.py --seed 42 --customers 500 --out <dir>`; its SHA-256 is checked):

```
python experiments/validation_exp3/run_experiment.py generate --out <data_dir>
python experiments/validation_exp3/run_experiment.py score    --data <data_dir> --train-data <seed42_dir> --scores <scores_dir>
python experiments/validation_exp3/run_experiment.py analyze  --scores <scores_dir>
python experiments/validation_exp3/verify_results.py --scores <scores_dir>
python experiments/validation_exp3/make_summary.py
```

## Notes on what differs from the plan or needs saying

* **Amendment A1** (committed before data existed) fixed the logistic-regression recipe, pooled bootstrap, first-fraud definition and that scores are committed only if under 25 MB. They are 54 MB, so they are not.
* **`--train-data` was added to the `score` command after the script was first committed**, because `data/v2` CSVs are not present in the repository; the command checks the directory's dataset hash against `data/v2/manifest.json` and refuses anything else. The scoring logic did not change.
* **Metadata change after the first analysis:** the first and second analysis runs recorded the SHA-256 of the gzip score files, which differs between runs (the gzip header holds a timestamp). The final run records the hash of the decompressed content. The three result files are identical in all three runs except for that field (`results/reproducibility_check.json`).
* **No fraud row in these datasets has fewer than 10 earlier transactions**, so nothing here tests the score or the model on fraud during the first transactions of a customer; the new-customer generator variant (seeds 511–515 style) was not used.
* The hold-out datasets and artifacts of earlier work (seeds 101–105, 201–205, 401–405, 411–415, 501–505, 511–515) and the saved models were only read, never scored again or changed; `git status` shows no change under `backend/models`.
* The comparison with the earlier hold-out results (seeds 501–505) is limited to ranking metrics of models that exist in both (default, LSTM → DNN, production): the datasets are different draws from the same generator with twice as many customers, and the alert cut-off rule differs, so alert counts are not compared.
