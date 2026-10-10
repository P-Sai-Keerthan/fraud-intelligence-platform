# Experiment 2 — review checklist (for approval)

Status: **review only. No model trained, no final data generated, no branch changed, nothing merged.**
Companion files: `AUDIT.md` (kept; two factual corrections marked), `PROTOCOL_DRAFT.md` (v0, unchanged),
`PROTOCOL_DRAFT_v1.md` (reviewed, awaiting approval).

## 0. Git state and merge plan (nothing below has been executed)

| Fact (checked read-only) | Value |
|---|---|
| Current branch | `claude/fair-sequence-fraud-detection-baiokj` at `cc0e1d0`, in sync with its remote; working tree clean; no stash |
| Commits unique to it | 1 (`cc0e1d0`, adds only `experiments/sequence_early_detection/*`) |
| `origin/Keerthan` | `5a3a7c8`, 14 commits ahead of `main` (294 files, +96,196/−880, 25.7 MB of new objects, **11 production source files modified**) |
| `origin/keerthan` (lower-case) | `372d3f3`, 3 commits ahead of `main`; **a different line of work** |
| Plain fast-forward to `Keerthan` | **No longer possible** (my branch has a commit `Keerthan` lacks). My earlier "fast-forward possible" was true only before `cc0e1d0` |
| Dry-run merge `Keerthan` → this branch | clean, no conflicts (`git merge-tree`, no ref or file touched) |
| Dry-run merge `keerthan` → this branch | clean |
| Dry-run `Keerthan` ⟷ `keerthan` | **conflicts** in ~30 files (`main.py`, `inference_pipeline.py`, models, schemas, frontend…) |
| Pull requests / other branches | none / none |

**Options.** Recommendation: **wait**, then do A once, after the ablation branch exists.

| | Plan | Risks |
|---|---|---|
| **A (recommended)** | One merge commit of the branch that holds both the v2 code and the ablation, into this branch | PR from this branch would show the whole `Keerthan` diff unless `Keerthan` lands in `main` first; `Keerthan` may move (re-merge later); production files appear in this branch (unchanged by me) |
| B | Rebase my one commit onto `Keerthan` | Rewrites `cc0e1d0` → needs a **force-push**; not recommended |
| C | No merge: read-only `git worktree add ../fraud-src origin/Keerthan`, experiment reads the generator by path and records the commit SHA | Less self-contained; path configuration |
| D | Vendor only `data/v2/synth_v2` and the split code into `experiments/…/vendor/` with SHA-256s | Code duplication; drift risk |

Exact steps for A, **to run only after you approve**:
```
git fetch origin
git tag backup-before-merge cc0e1d0                      # local safety tag
git merge --no-ff origin/<branch-with-v2-code-and-ablation> -m "Merge <branch> for Experiment 2"
git diff --stat cc0e1d0 HEAD -- experiments/sequence_early_detection   # must print nothing: my docs unchanged
git push origin claude/fair-sequence-fraud-detection-baiokj           # ordinary push, no force
```
Rollback: before pushing `git reset --hard backup-before-merge`; after pushing `git revert -m 1 <merge-sha>`.
**Open question for you:** which of `Keerthan` / `keerthan` is the line of record? They are not merge-compatible.
**Windows warning:** two branches differing only by letter case can collide in a clone on a case-insensitive file system.

## A. Where the previous experiment is, and what is missing

* Searched: container (`/mnt/attach`, `/mnt/user-data/*`, whole file system), every ref (`main`, `Keerthan`, `keerthan`, this
  branch; `git log --all -S"lstm_value_ablation"` finds only my own audit commit), the GitHub repository (4 branches, 0 PRs),
  and `.gitignore` on all three branches (nothing hides `experiments/`).
* Result: **`experiments/lstm_value_ablation/` exists nowhere I can read.** The attachments you named are Windows paths
  (`C:\Users\Keerthan\Downloads\REPORT.md`, `REPORT_1.md`, `lstm_value_ablation.zip`) that never reached this session.
* Needed (minimum): README.md, REPORT.md (and what `REPORT_1.md` is), the aggregate-feature code, split definition, the **list of
  data and training seeds**, per-seed result files, the decision rule text, library versions, dataset hashes or manifest.
* Restore: commit them to a **new** branch and push (commands in §9). Large arrays can stay out; keep a SHA-256 list.

## B. Validation of the previous experiment (blocked; nothing is verified)

| Check | Status |
|---|---|
| Aggregates use only information before the scored transaction (strict `<` windows, current row excluded, tie order, per-customer grouping) | **Unverified** — code unavailable |
| Future-perturbation and label-flip tests on the ablation's own feature function | **Unverified** — to run on arrival |
| Customer / transaction / target-label / temporal leakage (split type, component handling, scalers fitted on train only, OOF stacking folds) | **Unverified** |
| Seeds collide with mine or with `SPENT_SEEDS` | **Unverified** |
| Reported numbers reproduce from the ablation's own saved predictions/seeds | **Unverified.** I will **not** reconstruct results from the report text |
| Report claims | Everything stated in the task text about the ablation (LSTM > nine-feature baseline; aggregates > LSTM; no gain from adding LSTM) is **unverified claim** until the files are read |

## Phase 3 verdicts on `PROTOCOL_DRAFT.md` (v0)

| Requirement | Verdict | Evidence / residual risk |
|---|---|---|
| Primary endpoint = first-fraud recall at a prespecified FPR, validation-only thresholds | **Partly met** | Code reads correctly (`evaluate_scores`, `select_threshold`, `threshold_for_fpr` take only validation data). **Flaw:** ties let tie-heavy arms realise 0.37–0.67 % when asked for 1 % [simulated] → fixed in v1 (§7) |
| Fair tuning/training budget | **Partly met** | Equal trial counts, but v0 space ≈ 41 h [modelled], A1 "frozen vs tuned" conflict, T\* is a max of three → v1 §3/§6; residual: equal trials ≠ equal tuning effectiveness → best-so-far curves + extension rule |
| Comparable historical information | **Met for S2 vs A3 only** | LAG-K now ≡ RAW-SEQ; A2/A1 have less, S1 has less; roles stated |
| Final data untouched until frozen | **True now; mechanism weak in v0** | [measured] no data exist for reserved seeds. v0 froze rules, not outcomes → two-stage freeze. Residual: the generator is public and deterministic, so protection is procedural (git tags, ledger), not physical |
| CIs and paired comparisons appropriate | **Yes, with limits** | [simulated] coverage 93.0–96.3 %, false-claim 2.3 %; CI conditions on trained models; training-data variability excluded |
| Sample size justified | **Not in v0; explicit in v1** | assumptions listed; discordance *d* unmeasured → power gate at Freeze-2 |
| Generator B treated as synthetic generalisation | **Partly** | v0 text ok; ordering and wording rules added |
| First-fraud ≠ early warning; lead time separate | **Mostly** | v0 lead-time start was wrong (`precursor_start`) [measured: first observable signal comes a median 20 h later]; wording lint added |

## 1. Confirmed issues

1. Plain fast-forward to `Keerthan` impossible; `Keerthan` and `keerthan` conflict (git).
2. Ablation unreadable: not in container, any ref, any PR (searched).
3. Tie bias in FPR budget (simulated); `metrics.episode_metrics` calls first-fraud recall an "early-warning rate" (code).
4. `transaction_id` ≈ customer index (corr 0.9999) and device-id **format** separates fraud (15 % vs 69 % hashed ids) (measured on scratch seed 9).
5. v0 decision rule: power 0.52 at a true Δ equal to δ (simulated).
6. v0 tuning space ≈ 41 h; A3 memory extrapolates to ≈ 13 GB (measured + modelled).
7. v0 lead time overstated the opportunity (measured).
8. AUDIT.md: "17 commits ahead" (true: 14) and the fast-forward claim — corrected in place, history preserved.
9. Reassuring: generator is deterministic across environments — seed 42 regenerated here matches all five committed SHA-256 hashes.

## 2. Unresolved issues

* Ablation contents, aggregate definition, seeds, decision rule **[blocked]**.
* Discordance *d* between S2 and T\* (power input) — measurable only after tuning.
* Which branch is the line of record; merge/vendor choice **[USER]**.
* δ = 0.05 and the 1 % FPR budget are proposals **[USER]**.
* Generator B design acceptance **[USER]**; IEEE-CIS (needs files or allowed hosts) **[USER]**.
* Real GRU/tree training times and final-refit cost: only micro-benchmarks exist.
* Whether a bit-exact GRU determinism check holds on CPU with 3 threads (L11) — untested.
* Equal trial count may still favour an arm with a lower-dimensional space (mitigated, not removed).

## 3. Required fixes (status)

- [x] Two-stage freeze, seed-averaged primary, A1 split, LAG ≡ RAW-SEQ, forbidden ids, randomised cut-off, lead-time definition, three-way rule, cost-aware search space — written in v1.
- [ ] **You:** restore the ablation (§9) and decide §0 and the **[USER]** items.
- [ ] **Me, after approval:** implement `seqearly/` and the 20 tests (L1–L20) with **no real data touched**; run them on scratch seed 9 only.
- [ ] **Me:** DEV-only pilot to replace modelled times with measured ones.
- [ ] **Me:** validate the ablation per §B once available; update AGG and seeds; re-issue v1 as `PROTOCOL.md` for Freeze-1.

## 4. Metrics

Primary: first-fraud recall at FPR = 1 % (randomised, validation-fitted cut-off), S2 minus T\*, 5 FINAL datasets, paired stratified
cluster bootstrap. Secondary: recall at 0.1 % and F1 cut-off, PR-AUC, first-fraud PR-AUC, ROC-AUC, precision/recall/F1,
confusion matrices, false alerts per 1,000 legitimate, realised FPR, recall by position in episode, cold-start stratum,
R0 and A1-frozen for context. Exploratory (labelled): per-type recall, repeat victims, DEV time-split, permuted history,
without `failed_logins_24h`. Lead time: separate Task B, never Task A.

## 5. Required leakage and integrity tests (all before any final data)

L1 future-perturbation (bit-identical) · L2 label-independence · L3 forbidden columns · L4 strict order/ties · L5 F9 = production ·
L6 split and seed allow-list · L7 cut-off provenance (poisoned test labels) · L8 fit-on-train-only · L9 identical row set across arms ·
L10 one `first` row per episode · L11 determinism · L12 permuted-label sanity · L13 final-access ledger · L14 ID-relabelling
invariance · L15 `customer_id`/`transaction_id` absent · L16 randomised cut-off hits target on ties · L17 wording lint ·
L18 Freeze-2 hash check · L19 seed-42 regeneration matches manifest · L20 lazy windows = materialised windows.

## 6. Commands after approval (the CLI does not exist yet: step 0 creates and tests it on scratch seed 9 only)

```
cd experiments/sequence_early_detection
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt        # ~5.8 GB with PyTorch
python -m pytest tests -q                                                              # step 0: L1–L20 must pass
python -m seqearly.cli generate --role dev --seeds 7101 7102 7103 --customers 2000     # ~8 min/dataset, 4 in parallel
python -m seqearly.cli pilot-timing --role dev                                         # measured times, DEV only
python -m seqearly.cli tune --arm A0 A1 A2 A3 S1 S2 --trials 40 --seed 9001            # ~8 h modelled; resumable
python -m seqearly.cli refit --seeds 31 32 33 34 35
python -m seqearly.cli power-gate                                                      # d, Δ on DEV validation; may add 8106–8110
python -m seqearly.cli freeze2 && git add -A && git commit -m "Freeze-2" && git tag seq-early-selection-v1
python -m seqearly.cli generate --role final --seeds 8101 8102 8103 8104 8105 --customers 2000   # refuses without both tags
python -m seqearly.cli score-final                                                     # once; writes the access ledger
python -m seqearly.cli report && python -m seqearly.cli plots
```
Each command is resumable and writes JSON under `results/`; nothing is overwritten.

## 7. Runtime and disk

| Item | Value | Basis |
|---|---|---|
| Generate one 2,000-customer dataset | 30 s | **measured** (scratch seed 9) |
| Production feature builder, same dataset | 441 s, 1 core | **measured** |
| Dataset size / peak memory | 136 MB CSV / 728 MB | **measured** |
| Rows per dataset | 416,056 txns, 1,904 fraud, 440 episodes | **measured** (scratch) |
| GRU train / infer speed (3 threads) | 3.3 k–22 k / 9 k–65 k samples/s (torch); ≈ 0.4–0.6× that (TF-CPU) | **measured**, random tensors |
| RF / LightGBM cost | 2.0 s/tree at 400 k rows; 0.024 / 0.349 s/round (45 / 877 feats at 200 k rows) | **measured**, random matrices |
| Tuning wall-clock, v1 space | ≈ 8.3 h (lean 6.2 h; v0 space 41 h) | **modelled** from the above |
| Final refits, B repeat | not estimated | **unknown** until pilot |
| venv: PyTorch / TF-CPU | 5.8 GB / 1.8 GB | **measured** |
| Free disk at review | ≈ 19 GB; recommend ≥ 15 GB free | measured |
| Peak RAM, LightGBM 877 features | 3.4 GB at 200 k rows → ≈ 13 GB at 750 k (linear extrapolation) | measured + extrapolated |

## 8. The experiment must stop if

1. Any of L1–L20 fails, or a feature depends on a forbidden field.
2. A reserved seed's data exists before Freeze-2, or FINAL data are read before both tags, or a second scoring run is attempted.
3. Seeds collide with `SPENT_SEEDS` or with the ablation's seeds.
4. The ablation's own numbers cannot be reproduced from its files (report the gap; do not substitute).
5. The permuted-label sanity run gives validation ROC-AUC outside 0.45–0.55.
6. Any arm's realised DEV-validation FPR deviates > 20 % (relative) from its target after the randomised cut-off.
7. Populations differ between arms (L9), or scores are not reproducible (L11).
8. The power gate says N is inadequate and extra seeds are not approved.
9. A pilot shows > 2× the modelled time, RSS > 13 GB, or free disk < 5 GB: pause and ask.
10. A post-freeze change to protocol, code, search space, or seeds is wanted: record a deviation and ask first.
11. Any wording that claims early warning, real-world, or banking performance appears outside its allowed section.

## 9. What to do next in your terminal

```
# 1) Read-only: what exists, what is tracked, what is ignored (run in your local repo)
git status --short
git branch -a
git stash list
git log --oneline --all --decorate -8
git check-ignore -v experiments/lstm_value_ablation/REPORT.md      # a printed rule means it is ignored
git log --oneline --all -- experiments/lstm_value_ablation | head  # prints commits if it was ever committed

# 2) PowerShell: where is the folder and what are the biggest files (GitHub rejects files > 100 MB)
Get-ChildItem -Recurse -Directory -Filter lstm_value_ablation | Select-Object FullName
Get-ChildItem -Recurse -File experiments\lstm_value_ablation | Sort-Object Length -Descending | Select-Object -First 10 FullName,Length

# 3) Publish it on a NEW branch (does not touch Keerthan / keerthan; uncommitted files travel with you)
git switch -c experiments/lstm-value-ablation
git add experiments/lstm_value_ablation
git status --short                                                 # review what will be committed
git commit -m "Add completed lstm_value_ablation experiment"
git push -u origin experiments/lstm-value-ablation
# if it was already committed on some local branch X instead:  git push -u origin X:experiments/lstm-value-ablation
```
Then reply with: the branch name; the output of `git branch -a` (for the `Keerthan`/`keerthan` question); what `REPORT_1.md` is; and
your answers on the base-branch plan (§0), δ, the 1 % FPR budget, Generator B, IEEE-CIS, and PyTorch vs TF-CPU.
Do not merge or pull `Keerthan` and `keerthan` into each other locally in the meantime.
