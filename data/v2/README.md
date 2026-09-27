# Synthetic data v2

A deterministic generator for a more realistic synthetic dataset (Step 4C-2). **The application still uses the v1 data** (`data/transactions_with_features.csv`) and the v1 models. Nothing here changes `/predict`, the models or the frontend.

The design is in `claude/step4c2-synthetic-data-design.md` in the project notes. This README is the short version.

## Generate

```bash
python data/v2/generate.py                      # default: 500 customers, seed 42 -> data/v2/
python data/v2/generate.py --customers 60 --out /tmp/v2-sample --skip-features
python data/v2/sanity_report.py                 # audits data/v2 -> data/v2/DATA_SANITY_REPORT.md (~1.5 min)
```

The sanity report recomputes every check from the CSV files and exits with status 1 if it finds an anomaly.

The generator writes these files:

| File | Contents |
|---|---|
| `transactions.csv` | the v1 raw columns (same names, order and meaning), followed by the metadata columns |
| `transactions_with_features.csv` | the v1 features-file columns, followed by the metadata columns. It is built with the unchanged production `build_point_features`, which never sees the metadata. |
| `customers.csv` | segment, home city, household, devices and trips for each customer |
| `episodes.csv` | one row per fraud episode: type, ring, warning-period start, first and last fraud time |
| `login_failures.csv` | every failed login as a timestamped event (typo, forgot_password, credential_attack, fraud_session) |
| `manifest.json` | settings, counts, column lists and SHA-256 checksums. It records no wall-clock time and no output path, so a rerun in the same environment is byte-identical; the `versions` field (python / numpy / pandas) differs between environments, while the CSVs do not. |
| `DATA_SANITY_REPORT.md` | written by `sanity_report.py`: structure, class balance, warning-sign overlap, archetypes, rings, failed-login checks, leakage and giveaway checks, integrity checks |

The CSVs are not committed to git (`data/v2/.gitignore`). The same seed and settings reproduce them byte for byte, and `manifest.json` records their checksums.

## Fixed, reproducible inputs

- **Dates:** 2026-01-12 to 2026-07-06, the same span as v1. They come from `GeneratorConfig.start_date` and `days`, never from the clock.
- **Random draws:** every draw comes from a numpy generator keyed by (seed, domain, entity). Changing one customer or episode does not reshuffle the others.
- **Settings:** all of them live in `synth_v2/config.py`: customer segments, legitimate life events, the seven fraud archetypes and their probabilities, rings and warning periods.

## Metadata columns (never model features)

`merchant_id`, `network_id`, `fraud_type`, `fraud_episode_id`, `fraud_stage`, `fraud_ring_id`, `is_precursor`, `legit_context`, `customer_segment`.

`backend/app/features/ground_truth.py` lists them together with the `is_fraud` label. Feature engineering refuses to start if any of them is in `FEATURE_COLUMNS`, and the tests also check the DNN input columns and the model matrices.

## Labels

- **`is_fraud = 1`** only for unauthorized transactions, meaning the transactions generated for a fraud episode.
- **Warning period:** when an episode has one (`is_precursor = 1`), it contains only the customer's own legitimate transactions. These stay `is_fraud = 0`; credential-attack login failures show up in their `failed_logins_24h`.
- **Episode fields:** every episode has an explicit `fraud_episode_id` and a `fraud_stage` (`first` / `subsequent`). Ring episodes share a `fraud_ring_id`.
- **Evaluation grouping:** `synth_v2/groups.py` provides the keys later evaluation will need, so that a ring or a household never ends up across two splits.

## What differs from v1

- **Legitimate unusual behaviour:** travel, device upgrades, borrowed and household-shared devices, VPN and public networks, big and very small purchases, new categories, bursts and forgotten passwords. See `legit_context`.
- **Seven fraud archetypes:** account takeover, stolen card used online, small test charges followed by a cash-out, device/phone takeover, one large hit, fraud that resembles normal activity, and coordinated ring fraud. Each has probabilistic indicators, and none of them is certain.
- **Deliberate rings:** 3–6 victims per ring, with shared devices, networks and cash-out merchants used part of the time, all inside one attack wave of 3 days or less.
- **No `DEV_UNKNOWN_` ids:** every device that is not a customer's original `DEV_nnnn_A` / `_B` uses one neutral format (`DEV_7F3A91C2`), whether it belongs to a customer, a household or a fraudster.
- **`failed_logins_24h`:** now the number of login-failure events in `[t − 24h, t)`.
