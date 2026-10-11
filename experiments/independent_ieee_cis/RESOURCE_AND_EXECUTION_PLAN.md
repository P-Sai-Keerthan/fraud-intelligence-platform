# Resource and execution plan (estimates; nothing run, nothing downloaded)

## 1. Environment today (checked)

4 CPU cores, 15 GB RAM (13 GB free), **15 GB free disk**, no GPU; Python 3.13 environment with TensorFlow 2.21, scikit-learn 1.8, pandas 3.0; `lightgbm` and `torch` **not installed**. Outbound access to Kaggle (and several other hosts) is **blocked** by the environment's network policy. Chromium/Playwright irrelevant here.

## 2. Data volume (labels: [S]/[M] tags as in the compatibility report)

| Item | Estimate | Basis |
|---|---|---|
| Rows | 590,540 labelled | [S] |
| Raw CSV (transaction + identity) | several hundred MB | [M] |
| In memory after joining, float32 | 1–2 GB; peak during feature building 3–4 GB | modelled |
| Cached features and sequences (lazy) | < 1 GB | design |
| Per-row scores for all arms and seeds | < 300 MB compressed | design |
| Virtual environment (TensorFlow-CPU + LightGBM) | about 2 GB | measured in Experiment 2 (TF-CPU venv 1.8 GB) |

Disk need: ≥ 8 GB free recommended; the current 15 GB is enough.

## 3. Time (modelled from Experiment 2 micro-benchmarks and the Experiment 3 run; to be replaced by a measured pilot on stage-3 data)

| Step | Estimate |
|---|---|
| Stage-0 audit, entity diagnostics | minutes |
| Vectorised feature build | < 10 min (the existing per-row builder would need ≈ 11 min per 590 k rows, so vectorising is for testability, not necessity) |
| Tuning, 6 learned arms × 40 trials on 3 CPU threads | 6–10 h (Experiment 2's modelled 8.3 h for a 750 k-row problem; this one is smaller but has more categorical handling) |
| Final refits (5 seeds × 6 arms) | 2–4 h |
| Final scoring, bootstrap (2,000 replicates × arms) | < 1 h (Experiment 3's pooled bootstrap over 620 k rows took about 12 min for nine models) |
| Total compute | roughly 1–1.5 working days wall-clock, resumable per stage |

## 4. Execution plan

| Step | Who | Output |
|---|---|---|
| 0 | owner | read and accept the dataset rules; decide D1–D6 (`DECISIONS_FOR_APPROVAL.md`) |
| 1 | owner | download locally and place files under an agreed path **or** add the Kaggle host to the environment's allowed domains (the owner downloads; this session does not accept terms) |
| 2 | assistant | record SHA-256 of the raw files; run stage-0 audit (label-blind); write `AUDIT0.json`; stop and report |
| 3 | owner + assistant | review the audit; revise the protocol where an [M] item was wrong |
| 4 | assistant | implement the package and tests IL1–IL22 on a **synthetic miniature** with the same schema (no real data); commit |
| 5 | assistant | entity diagnostics (label-blind) → `ENTITY.json`; Freeze-1 tag |
| 6 | assistant | stage-3 development on train/validation labels; pilot timing |
| 7 | assistant | Freeze-2 tag (selected configs, hashes, feasibility gate) |
| 8 | assistant | final scoring once; report; ledger |

Stop and ask at: end of step 2, step 5, step 7.

## 5. Risks to the plan

Network or licence blocks the download · stage 0 shows no usable entity (answer: report that the sequence question is not answerable here) · memory spikes with 400+ columns (select columns by role first, cache as parquet) · LightGBM or TensorFlow version drift (pin and record) · the final segment's fraud count is too small for the first-fraud analysis (descriptive only).
