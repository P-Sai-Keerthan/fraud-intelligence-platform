# Review evidence (not results)

Everything here supports statements in `PROTOCOL_DRAFT_v1.md` and `REVIEW_CHECKLIST.md`. **None of it is a result about
fraud detection or about any model.** It uses scratch seed 9, the committed seed-42 dataset, random tensors/matrices,
or purely simulated paired outcomes. No reserved development or final seed (7101–7103, 8101–8105, 7201–7203, 8201–8205)
was generated.

| Script (`scripts/`) | What it checks | Output (`output/`) |
|---|---|---|
| `sim_stats.py <sims> <reps>` | Coverage of the paired cluster bootstrap, false-claim rate, power of the decision rules (simulated outcomes only) | `sim_stats_300sims_400reps.jsonl` |
| `struct_checks.py <dataset_dir>` | `transaction_id` order, device-id format vs fraud, warning-period timing, on scratch seed 9 | `struct_checks_scratch_seed9.txt` |
| `tie_check.py <repo_src_dir>` | Existing cut-off functions on tie-heavy synthetic scores | `tie_check.txt` |
| `randomized_cutoff_proto.py` | Prototype of the validation-fitted randomised cut-off | `randomized_cutoff_proto.txt` |
| `verify_determinism.py <manifest> <dir>` | Regenerated seed 42 vs committed SHA-256 hashes | `determinism_seed42.txt` |
| `measure_gen.py <src> <out> <customers> <seed>` | Generation and feature-building time, memory, disk | `generation_timing_2000_customers_scratch_seed9.json` |
| `bench_dl.py torch\|tf` | GRU throughput on random tensors | `bench_gru_torch.json`, `bench_gru_tf.json` |
| `bench_trees.py` | RF / LightGBM throughput on random matrices | `bench_trees.txt` (console lines transcribed verbatim) |
| `cost_model.py` | Tuning wall-clock model built from the benchmarks | `cost_model.txt` |

Paths are arguments; the generator source used was `origin/Keerthan` at `5a3a7c8`. Environment: `output/versions.txt`.
Timing numbers come from a 4-core container with 3 threads and one other job running at times, so they are indicative.
`cost_model.txt` is a **model**, not a measurement of a full run.
