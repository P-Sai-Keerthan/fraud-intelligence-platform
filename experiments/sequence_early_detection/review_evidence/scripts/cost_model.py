"""Cost model from MEASURED micro-benchmarks (random data; 3 threads). Not an end-to-end measurement. Assumptions are printed."""
import numpy as np
N_ALL_TXN_DEV = 3 * 416_056                    # measured: scratch 2000-customer dataset has 416,056 transactions
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
TRAIN_ROWS = int(N_ALL_TXN_DEV * TRAIN_FRAC); VAL_ROWS = int(N_ALL_TXN_DEV * VAL_FRAC)
POS = int(3 * 1904 * TRAIN_FRAC)               # measured fraud rows per scratch dataset
print(f"assumed DEV-train rows {TRAIN_ROWS:,} (positives {POS:,}); DEV-validation rows {VAL_ROWS:,}")
ratio_rows = {"1:10": POS * 11, "1:50": POS * 51, "all": TRAIN_ROWS}
# --- GRU: measured train/infer samples per second for (hidden,layers); 2-layer h=32 interpolated as half of 1-layer
train_sps = {(32,1):22487,(64,1):11186,(128,1):6780,(32,2):11200,(64,2):5645,(128,2):3325}
infer_sps = {(32,1):65499,(64,1):38338,(128,1):18021,(32,2):32700,(64,2):22655,(128,2):9397}
def gru_trial_h(ratio, epochs, val_frac=1.0):
    t = []
    for k in train_sps: t.append(epochs * (ratio_rows[ratio] / train_sps[k] + VAL_ROWS * val_frac / infer_sps[k]))
    return np.mean(t) / 3600
# --- RF: 2.0 s/tree at 400k rows (3 threads), scaling ~ n^1.25 (measured 100k->400k: 0.35 -> 1.99 s/tree)
rf_tree = lambda n: 1.99 * (n / 400_000) ** 1.25
rf_trial_h = lambda ratio: 450 * rf_tree(ratio_rows[ratio]) / 3600          # n_estimators in {300, 600}: mean 450
# --- LightGBM: 0.024 s/round at 200k x 45 ; 0.349 s/round at 200k x 877 (linear in rows); mean 400 rounds after early stopping
lgb_round = lambda n, d: (0.024 if d <= 45 else 0.349) * n / 200_000
lgb_trial_h = lambda ratio, d: 400 * lgb_round(ratio_rows[ratio], d) / 3600
rows = []
for ratio in ("1:10", "1:50", "all"):
    rows.append((ratio, f"{ratio_rows[ratio]:,}", rf_trial_h(ratio), lgb_trial_h(ratio, 45), lgb_trial_h(ratio, 877),
                 gru_trial_h(ratio, 12, 0.25), gru_trial_h(ratio, 25, 0.25)))
print("\nhours PER TRIAL by negative-sampling ratio (mean over sampled hyper-parameters):")
print(f'{"neg ratio":>9} {"rows/epoch":>10} | {"RF A0/A1*":>9} {"LGBM A2":>8} {"LGBM A3":>8} | {"GRU 12ep":>8} {"GRU 25ep":>8}   (*RF cost for 45 features; A0 with 9 features ~0.4x)')
for r in rows: print(f"{r[0]:>9} {r[1]:>10} | {r[2]:>9.2f} {r[3]:>8.3f} {r[4]:>8.2f} | {r[5]:>8.2f} {r[6]:>8.2f}")
def budget(trials, ratios, gru_epochs):
    m = lambda f: np.mean([f(r) for r in ratios])
    a1 = trials * m(rf_trial_h); a0 = trials // 2 * m(rf_trial_h) * 0.4
    a2 = trials * m(lambda r: lgb_trial_h(r, 45)); a3 = trials * m(lambda r: lgb_trial_h(r, 877))
    gru = trials * m(lambda r: gru_trial_h(r, gru_epochs, 0.25))
    return dict(A0=a0, A1=a1, A2=a2, A3=a3, S1=gru, S2=gru * 1.1)
print("\nTOTAL tuning wall-clock (hours, 3 threads, one trial at a time), by budget variant:")
for name, trials, ratios, ep in [("v0 as drafted: 40 trials, ratios {10,50,all}, GRU <=25 epochs", 40, ("1:10","1:50","all"), 25),
                                 ("proposed: 40 trials, ratios {10,50}, GRU search cap 12 epochs", 40, ("1:10","1:50"), 12),
                                 ("lean: 30 trials, ratios {10,50}, GRU search cap 12 epochs", 30, ("1:10","1:50"), 12)]:
    b = budget(trials, ratios, ep); tot = sum(b.values())
    print(f"  {name}\n    " + "  ".join(f"{k} {v:5.1f}" for k, v in b.items()) + f"   | TOTAL {tot:5.1f} h")
