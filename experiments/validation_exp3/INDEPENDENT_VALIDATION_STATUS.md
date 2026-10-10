# Independent validation — what exists, what is missing, what a valid test needs (B6)

Status at the time of writing: **no independent or real-world fraud dataset is available to this project, and none was downloaded.**
Every number in the repository, including Experiment 3, comes from the project's own synthetic generators.

## 1. What was checked

* Searched the repository and the working machine for the usual public fraud data (credit-card fraud, IEEE-CIS, PaySim, simulated card data, TabFormer and similar) by file name. (Files with unrelated names were not opened.) Nothing was found. The repository's documents mention such datasets only as suggestions.
* Tracked data files are all generated: the v1 seed CSV (`data/`), the v2 score files and three small `test_data` CSVs.
* No download was attempted. Any download needs the owner's authorisation, a licence check, and a data-handling decision (see section 4).

## 2. Why the synthetic hold-outs are not independent validation

| Evidence in the repository | What it can show | What it cannot show |
|---|---|---|
| v1 evaluation (near-perfect scores) | the model can learn the v1 generator's rule | anything about real fraud: v1 fraud is almost trivially separable |
| v2 time and customer hold-outs (seeds 101–105, 201–205, 401–405, 411–415, 501–505, 511–515) | the models work on new draws of the **same generator**, with fraud patterns the authors wrote | that the patterns resemble real fraud; that the features are the ones fraudsters leave behind; behaviour under label noise, concept drift or adversaries |
| Experiment 3 (seeds 9301–9305) | the same, plus similarity, calibration and ablations on the same generator | the same limits |

A new seed changes the random draws, not the assumptions. The generator's fraud types, feature definitions and customer behaviour are the same, so a model that has learnt them scores well on every seed. This is **in-distribution synthetic validation**, not external validation.

## 3. Evidence that would be needed

1. **At least one dataset not produced by this project's generator**, with real or at least independently simulated fraud labels. Preferred order: (a) real transactions from a partner institution under an agreement, (b) a public real-world set with a clear licence, (c) an independent simulator written by someone else.
2. Required characteristics: per-customer transaction history with timestamps; an account or customer identifier that allows past-only features; amount, merchant category, device or channel, location (or substitutes); a fraud label with a known labelling delay and definition (confirmed fraud versus chargeback); at least a few hundred fraud cases in the evaluation period (the v2 time hold-out had 61, which gave wide intervals); a stated fraud rate.
3. Feature mapping: how each of the nine behavioural features (`amount_zscore`, `hour_is_unusual`, `is_new_device`, `is_new_location`, `is_foreign_location`, `failed_logins_24h`, `category_is_unusual`, `txn_velocity_1h`, `amount_pct_of_avg`) is computed from the new schema. Several need fields many public sets do not have (device id, failed logins). Where a field is missing the feature must be reported as unavailable rather than filled with zeros, and the model re-evaluated or re-trained with that restriction, which is a different model from the current one.

## 4. Leakage risks to control

* **Temporal leakage:** features must use only earlier rows (the project's feature code does; the new mapping must keep it). Use time-ordered splits, with the final period untouched until the end.
* **Customer leakage:** the same customer or household in training and evaluation; use customer-grouped splits.
* **Label leakage:** columns recorded after the fraud was found (chargeback flags, blocked status, case ids) must not be inputs; the repository's `assert_no_ground_truth` guard should be extended to the new schema's names.
* **Duplicate and near-duplicate rows** across splits (common in public sets built by oversampling).
* **Threshold and calibration selection** on the final period: choose them on a separate validation period.
* **Class-balance artefacts:** public sets are often down-sampled; precision and alert volume do not transfer to a production rate unless re-weighted.
* **Privacy and licence:** personal data cannot be placed in this repository, in reports or in the PDF; licence terms may restrict redistribution or publication.

## 5. Final evaluation procedure (to be fixed before the data is opened)

1. Write the protocol first (hypotheses, population, metrics, cut-off rule, models, comparison baselines, analysis of subgroups) and commit it.
2. Map the schema and compute features with the existing past-only code; record the mapping and a sample audit.
3. Split by time and by customer group: training, validation (thresholds, calibration), and a final period used **once**.
4. Evaluate the saved default model **without re-training** first (transfer), then, separately, a model re-trained on the new data with the same recipe. Report both.
5. Metrics: PR-AUC and ROC-AUC with customer-cluster bootstrap intervals, precision, recall and false-positive rate at a cut-off fixed on validation, alerts per 1,000 transactions, first-fraud recall, calibration (Brier, log loss, reliability), and the comparison with the simple baselines (rule, logistic regression, forest without sequence input).
6. Report negative results in full. The decision "does the sequence model add value" is made on the paired difference interval against the no-sequence baseline.

## 6. Decisions needed from the owner

* Whether a partner dataset or a public dataset will be used, and who signs the data-use terms.
* Whether an independent simulator should be commissioned as an interim step (still synthetic, but not self-authored).
* The label definition and delay that match the intended deployment.
