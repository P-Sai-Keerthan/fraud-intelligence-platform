"""Structural checks of generator behaviour on SCRATCH seed 9 (500 customers). No model is trained or scored."""
import re, sys, numpy as np, pandas as pd
D = sys.argv[1]
tx = pd.read_csv(f"{D}/transactions.csv", keep_default_na=False)
ep = pd.read_csv(f"{D}/episodes.csv", keep_default_na=False)
lg = pd.read_csv(f"{D}/login_failures.csv")
for c in ("timestamp",): tx[c] = pd.to_datetime(tx[c]); lg[c] = pd.to_datetime(lg[c])

print("== 1. transaction_id / customer_id carry identity and order (so they must never be features)")
tid = tx["transaction_id"].str[4:].astype(int); cid = tx["customer_id"].str[5:].astype(int)
print("rows:", len(tx), "| transaction_id strictly increasing in file order:", bool((tid.diff().dropna() > 0).all()),
      "| file order is (customer, time):", bool(((cid.diff().dropna() >= 0)).all()))
print("Spearman-like: corr(transaction_id, customer index) =", round(float(np.corrcoef(tid, cid)[0, 1]), 4))

print("\n== 2. device-id STRING FORMAT vs fraud (hazard if ids were used as raw categories)")
orig = tx["device_id"].str.match(r"^DEV_\d{4}_[AB]$")
print("share of rows whose device id looks like DEV_dddd_[AB]:  legit %.4f | fraud %.4f" % (orig[tx.is_fraud == 0].mean(), orig[tx.is_fraud == 1].mean()))
print("share of rows whose device id is a 'neutral hash' DEV_XXXXXXXX: legit %.4f | fraud %.4f" % ((~orig)[tx.is_fraud == 0].mean(), (~orig)[tx.is_fraud == 1].mean()))
own_numeric = tx["device_id"].str.extract(r"^DEV_(\d{4})_[AB]$")[0]
print("original-format device carries the customer's OWN index in %.1f%% of such rows (rest: another customer's device)" % (100 * (own_numeric.dropna().astype(int) == cid[own_numeric.notna()]).mean()))

print("\n== 3. warning period: when is the first OBSERVABLE signal, versus precursor_start?")
w = ep[ep["precursor_start"] != ""].copy()
w["precursor_start"] = pd.to_datetime(w["precursor_start"]); w["first_fraud_time"] = pd.to_datetime(w["first_fraud_time"])
rows = []
for _, e in w.iterrows():
    ev = lg[(lg.customer_id == e.customer_id) & (lg.timestamp >= e.precursor_start) & (lg.timestamp < e.first_fraud_time) & (lg.source == "credential_attack")]
    t_sig = ev.timestamp.min() if len(ev) else pd.NaT
    t = tx[(tx.customer_id == e.customer_id) & (tx.is_fraud == 0)]
    in_w = t[(t.timestamp >= e.precursor_start) & (t.timestamp < e.first_fraud_time)]
    after = in_w[in_w.timestamp >= t_sig] if pd.notna(t_sig) else in_w.iloc[0:0]
    rows.append(dict(ep=e.fraud_episode_id, type=e.fraud_type, n_attack_events=len(ev),
                     gap_signal_after_start_h=(t_sig - e.precursor_start).total_seconds() / 3600 if pd.notna(t_sig) else np.nan,
                     window_h=(e.first_fraud_time - e.precursor_start).total_seconds() / 3600,
                     lead_from_signal_h=(e.first_fraud_time - t_sig).total_seconds() / 3600 if pd.notna(t_sig) else np.nan,
                     txns_in_window=len(in_w), txns_after_first_signal=len(after),
                     after_with_failed_login=int((after.failed_logins_24h > 0).sum())))
r = pd.DataFrame(rows)
print("episodes with a warning period:", len(r), "| by type:", r.type.value_counts().to_dict())
print("episodes with >=1 credential_attack event inside the window:", int((r.n_attack_events > 0).sum()))
print("gap between precursor_start and first observable attack event (hours): median %.1f, mean %.1f, max %.1f" % (r.gap_signal_after_start_h.median(), r.gap_signal_after_start_h.mean(), r.gap_signal_after_start_h.max()))
print("nominal window (precursor_start -> first fraud) hours: median %.1f | lead measured from first observable signal: median %.1f" % (r.window_h.median(), r.lead_from_signal_h.median()))
print("legit transactions in nominal window: total %d | AFTER the first observable signal: total %d" % (r.txns_in_window.sum(), r.txns_after_first_signal.sum()))
print("episodes with at least one transaction after the first signal (= an opportunity to alert at all): %d of %d" % (int((r.txns_after_first_signal > 0).sum()), len(r)))
print("of those post-signal transactions, those with failed_logins_24h>0: %d of %d" % (r.after_with_failed_login.sum(), r.txns_after_first_signal.sum()))
