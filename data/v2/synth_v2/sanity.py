"""
Data sanity audit of a generated v2 dataset.

    python data/v2/sanity_report.py            # audits data/v2, writes data/v2/DATA_SANITY_REPORT.md

`audit(directory)` recomputes everything from the CSV files (it does not trust
the generator's own bookkeeping) and returns a results dict; `render(results)`
turns it into Markdown. `anomalies` lists every failed check; an empty list
means no generator bug was found. The report contains no wall-clock time, so
it is reproducible from the data.
"""

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema
from .config import ARCHETYPES, DOMESTIC_CITIES, FOREIGN_CITIES, FOREIGN_CITIES_V1, MERCHANT_CATEGORIES
from .generator import DAY, count_preceding
from .output import FILES, MANIFEST

BACKEND_DIR = Path(__file__).resolve().parents[3] / "backend"
ORIGINAL_DEVICE = re.compile(r"^DEV_(\d{4})_[AB]$")
NEUTRAL_DEVICE = re.compile(r"^DEV_[0-9A-F]{8}$")
NEUTRAL_NETWORK = re.compile(r"^NET_[0-9A-F]{8}$")
MERCHANT = re.compile(r"^MER_\d{5}$")

# warning signs computed from the PRODUCTION feature columns
SIGNALS = {
    "unusual hour": lambda f: f["hour_is_unusual"] == 1,
    "new device": lambda f: f["is_new_device"] == 1,
    "foreign location": lambda f: f["is_foreign_location"] == 1,
    "unusual category": lambda f: f["category_is_unusual"] == 1,
    "failed logins in previous 24h (>=1)": lambda f: f["failed_logins_24h"] >= 1,
    "another transaction within the hour": lambda f: f["txn_velocity_1h"] >= 1,
    "amount >= 300% of customer average": lambda f: f["amount_pct_of_avg"] >= 300,
}
EXTRA_SIGNALS = {
    "new location": lambda f: f["is_new_location"] == 1,
    "failed logins in previous 24h (>=2)": lambda f: f["failed_logins_24h"] >= 2,
    "amount >= 188% of customer average": lambda f: f["amount_pct_of_avg"] >= 188,
}


def _pct(x) -> float:
    return round(float(np.mean(x)) * 100, 2) if len(x) else float("nan")


def _production_feature_lists():
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from app.features import feature_engineering as fe
    from app.features.ground_truth import FORBIDDEN_FEATURE_COLUMNS
    # DNN_INPUT_COLUMNS is read from source so the audit does not need TensorFlow
    tree = ast.parse((BACKEND_DIR / "app" / "models" / "dnn_model.py").read_text())
    dnn_cols = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "DNN_INPUT_COLUMNS")
    return fe, list(fe.FEATURE_COLUMNS), dnn_cols, FORBIDDEN_FEATURE_COLUMNS


def load(directory):
    d = Path(directory)
    tx = pd.read_csv(d / FILES["transactions"], keep_default_na=False)
    feat = pd.read_csv(d / FILES["features"], keep_default_na=False)
    cust = pd.read_csv(d / FILES["customers"], keep_default_na=False)
    ep = pd.read_csv(d / FILES["episodes"], keep_default_na=False)
    lf = pd.read_csv(d / FILES["login_failures"], keep_default_na=False)
    manifest = json.loads((d / MANIFEST).read_text())
    return tx, feat, cust, ep, lf, manifest


def audit(directory, recompute_features: bool = True) -> dict:
    tx, feat, cust, ep, lf, manifest = load(directory)
    anomalies = []

    def check(ok, message):
        if not bool(ok):
            anomalies.append(message)
        return bool(ok)

    R = {"anomalies": anomalies}
    tx["ts"] = pd.to_datetime(tx["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    feat["ts"] = pd.to_datetime(feat["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    lf["ts"] = pd.to_datetime(lf["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    cfg = manifest["config"]
    start = pd.Timestamp(cfg["start_date"])
    end = start + pd.Timedelta(days=cfg["days"])
    fraud = tx["is_fraud"] == 1
    legit = ~fraud

    # ---- files
    R["files"] = {name: {"rows": info["rows"], "sha256": info["sha256"],
                         "bytes": (Path(directory) / name).stat().st_size}
                  for name, info in manifest["files"].items()}
    check(all(R["files"][n]["rows"] == len(t) for n, t in
              ((FILES["transactions"], tx), (FILES["features"], feat), (FILES["customers"], cust),
               (FILES["episodes"], ep), (FILES["login_failures"], lf))), "manifest row counts do not match the files")

    # ---- structure
    ring_eps = ep[ep["fraud_ring_id"] > 0]
    ring_sizes = ring_eps.groupby("fraud_ring_id")["customer_id"].nunique()
    R["structure"] = {
        "transactions": len(tx), "customers": tx["customer_id"].nunique(),
        "first_timestamp": str(tx["ts"].min()), "last_timestamp": str(tx["ts"].max()),
        "configured_range": f"{start.date()} to {(end - pd.Timedelta(days=1)).date()} ({cfg['days']} days)",
        "fraud": int(fraud.sum()), "fraud_pct": _pct(fraud),
        "legit": int(legit.sum()), "legit_pct": _pct(legit),
        "episodes": len(ep), "episodes_by_type": ep["fraud_type"].value_counts().sort_index().to_dict(),
        "ring_episodes": len(ring_eps), "rings": int(ring_sizes.size),
        "ring_sizes": ring_sizes.value_counts().sort_index().to_dict(),
        "warning_period_transactions": int(tx["is_precursor"].sum()),
        "episodes_with_warning_period": int((ep["precursor_start"] != "").sum()),
        "households": int(cust.loc[cust["household_id"] > 0, "household_id"].nunique()),
        "customers_in_households": int((cust["household_id"] > 0).sum()),
    }

    # ---- class balance
    month = tx["ts"].dt.strftime("%Y-%m")
    R["balance"] = {
        "by_segment": tx.groupby("customer_segment").agg(
            customers=("customer_id", "nunique"), transactions=("is_fraud", "size"),
            fraud=("is_fraud", "sum"), fraud_pct=("is_fraud", lambda s: _pct(s))).reset_index(),
        "by_type": tx[fraud].groupby("fraud_type").agg(fraud=("is_fraud", "size")).assign(
            pct_of_all_transactions=lambda d: (d["fraud"] / len(tx) * 100).round(3),
            pct_of_fraud=lambda d: (d["fraud"] / fraud.sum() * 100).round(1)).reset_index(),
        "by_month": tx.assign(month=month).groupby("month").agg(
            transactions=("is_fraud", "size"), fraud=("is_fraud", "sum"),
            fraud_pct=("is_fraud", lambda s: round(float(s.mean()) * 100, 3)),
            episodes_starting=("fraud_stage", lambda s: int((s == "first").sum()))).reset_index(),
    }

    # ---- warning-sign overlap (production features)
    fraud_f = feat["is_fraud"] == 1
    pre_f = feat["is_precursor"] == 1
    rows = []
    for name, fn in {**SIGNALS, **EXTRA_SIGNALS}.items():
        flag = fn(feat)
        rows.append({"signal": name, "fraud_pct": _pct(flag[fraud_f]), "legit_pct": _pct(flag[~fraud_f]),
                     "legit_warning_period_pct": _pct(flag[pre_f]),
                     "legit_outside_warning_pct": _pct(flag[~fraud_f & ~pre_f])})
        if name in SIGNALS:
            check(flag[fraud_f].any() and flag[~fraud_f].any(), f"signal '{name}' occurs in only one class")
            check(not (flag == fraud_f).all(), f"signal '{name}' identifies fraud perfectly")
    n_signals = np.sum([fn(feat).to_numpy() for fn in SIGNALS.values()], axis=0)
    feat["_n_signals"] = n_signals
    dist = pd.cut(pd.Series(n_signals[fraud_f.to_numpy()]), [-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"])
    R["overlap"] = {
        "signals": pd.DataFrame(rows),
        "fraud_signal_count": dist.value_counts().reindex(["0", "1", "2", "3+"]).to_dict(),
        "legit_signal_count": pd.cut(pd.Series(n_signals[~fraud_f.to_numpy()]), [-1, 0, 1, 2, 99],
                                     labels=["0", "1", "2", "3+"]).value_counts().reindex(["0", "1", "2", "3+"]).to_dict(),
    }
    fe, feature_cols, dnn_cols, forbidden = _production_feature_lists()
    separable = []
    for col in feature_cols:
        f, l = feat.loc[fraud_f, col], feat.loc[~fraud_f, col]
        if f.min() > l.max() or f.max() < l.min():
            separable.append(col)
    check(not separable, f"features that separate the classes by a single threshold: {separable}")
    R["overlap"]["single_threshold_separable"] = separable
    rule = (feat["hour_is_unusual"] == 1) & (feat["amount_pct_of_avg"] >= 188)
    R["v1_rule"] = {
        "flagged": int(rule.sum()), "true_positives": int((rule & fraud_f).sum()),
        "precision": round(float(feat.loc[rule, "is_fraud"].mean()), 4) if rule.any() else float("nan"),
        "recall": round(float(rule[fraud_f].mean()), 4),
    }
    check(not (R["v1_rule"]["precision"] == 1 and R["v1_rule"]["recall"] == 1), "v1 rule is still perfect")

    # ---- legitimate unusual behaviour
    ctx = tx.loc[tx["legit_context"] != "none", ["customer_id", "is_fraud", "legit_context"]].copy()
    ctx["context"] = ctx["legit_context"].str.split("|")
    ctx = ctx.explode("context")
    R["legit_unusual"] = ctx.groupby("context").agg(
        rows=("is_fraud", "size"), customers=("customer_id", "nunique"), fraud_rows=("is_fraud", "sum")
    ).reindex(list(schema.LEGIT_CONTEXTS)).fillna(0).astype(int).reset_index()
    check((R["legit_unusual"]["fraud_rows"] == 0).all(), "a legitimate-context row is labelled fraud")
    check((R["legit_unusual"]["rows"] > 0).all(), "some legitimate context never occurs")
    legit_foreign = tx[legit & tx["location"].isin(FOREIGN_CITIES)]
    R["legit_extra"] = {
        "foreign_rows": len(legit_foreign), "foreign_customers": legit_foreign["customer_id"].nunique(),
        "rows_with_failed_logins": int((legit & (tx["failed_logins_24h"] > 0)).sum()),
        "customers_with_upgrade": int((cust["upgrade_device"] != "").sum()),
        "customers_with_secondary_device": int((cust["secondary_device"] != "").sum()),
    }

    # ---- fraud archetypes
    arch_rows = []
    for a in ARCHETYPES:
        eids = set(ep.loc[ep["fraud_type"] == a, "fraud_episode_id"])
        fr = feat[fraud_f & (feat["fraud_type"] == a)]
        pre_n = 0
        for _, e in ep[ep["fraud_episode_id"].isin(eids) & (ep["precursor_start"] != "")].iterrows():
            m = ((tx["customer_id"] == e["customer_id"]) & (tx["is_precursor"] == 1)
                 & (tx["ts"] >= pd.Timestamp(e["precursor_start"])) & (tx["ts"] < pd.Timestamp(e["first_fraud_time"])))
            pre_n += int(m.sum())
        row = {"archetype": a, "episodes": len(eids), "fraud_rows": len(fr), "warning_period_rows": pre_n}
        for name, fn in SIGNALS.items():
            row[name] = _pct(fn(fr))
        row["zero_signals_pct"] = _pct(fr["_n_signals"] == 0)
        row["median_amount_pct_of_avg"] = round(float(fr["amount_pct_of_avg"].median()), 1) if len(fr) else float("nan")
        arch_rows.append(row)
        check(len(eids) > 0 and len(fr) > 0, f"archetype {a} has no episodes or no fraud rows")
    R["archetypes"] = pd.DataFrame(arch_rows)

    # signature check: no archetype has one deterministic device/location/category pattern
    fr_tx = tx[fraud].copy()
    fr_tx["device_kind"] = np.where(fr_tx["device_id"].str.match(ORIGINAL_DEVICE), "own original device", "other device")
    fr_tx["location_kind"] = np.where(fr_tx["location"].isin(FOREIGN_CITIES), "foreign", "domestic")
    sig = []
    for a, g in fr_tx.groupby("fraud_type"):
        combo = g.groupby(["device_kind", "location_kind", "merchant_category"]).size()
        sig.append({"archetype": a, "distinct_device_location_category_combos": int(combo.size),
                    "largest_combo_share_pct": round(float(combo.max() / len(g) * 100), 1),
                    "categories": int(g["merchant_category"].nunique()), "locations": int(g["location"].nunique())})
        check(combo.size > 1, f"archetype {a} has a single device/location/category signature")
    R["signatures"] = pd.DataFrame(sig)

    # ---- rings
    ring_tx = tx[tx["fraud_ring_id"] > 0]
    legit_devices = set(tx.loc[legit, "device_id"])
    legit_networks = set(tx.loc[legit, "network_id"])
    ring_rows = []
    for rid, g in ring_tx.groupby("fraud_ring_id"):
        shared = {}
        for col in ("device_id", "network_id", "merchant_id"):
            per = g.groupby(col)["customer_id"].nunique()
            shared[col] = set(per[per >= 2].index)
        uses_shared = g["device_id"].isin(shared["device_id"]) | g["network_id"].isin(shared["network_id"]) \
            | g["merchant_id"].isin(shared["merchant_id"])
        firsts = g.groupby("customer_id")["ts"].min().sort_values()
        span = g["ts"].max() - g["ts"].min()
        ring_rows.append({
            "ring": int(rid), "victims": g["customer_id"].nunique(), "fraud_rows": len(g),
            "attack_window_hours": round(span.total_seconds() / 3600, 1),
            "max_gap_between_victims_hours": round(firsts.diff().max().total_seconds() / 3600, 1) if len(firsts) > 1 else 0.0,
            "shared_devices": len(shared["device_id"]), "shared_networks": len(shared["network_id"]),
            "shared_merchants": len(shared["merchant_id"]),
            "rows_using_shared_device_pct": _pct(g["device_id"].isin(shared["device_id"])),
            "rows_using_shared_network_pct": _pct(g["network_id"].isin(shared["network_id"])),
            "rows_using_shared_merchant_pct": _pct(g["merchant_id"].isin(shared["merchant_id"])),
            "rows_using_any_shared_pct": _pct(uses_shared),
            "shared_devices_also_used_legitimately": len(shared["device_id"] & legit_devices),
            "shared_networks_also_used_legitimately": len(shared["network_id"] & legit_networks),
        })
        check(span <= pd.Timedelta(hours=cfg["ring_wave_max_hours"]), f"ring {rid} spans more than one attack wave")
        check(any(shared.values()), f"ring {rid} has no shared infrastructure")
        members = set(ep.loc[ep["fraud_ring_id"] == rid, "customer_id"])
        check(set(g["customer_id"]) == members, f"ring {rid} rows and episode table disagree on members")
    R["rings"] = pd.DataFrame(ring_rows)
    dev_c = tx.groupby("device_id").agg(customers=("customer_id", "nunique"), fraud=("is_fraud", "max"),
                                        ring=("fraud_ring_id", "max"))
    net_c = tx[legit].groupby("network_id")["customer_id"].nunique()
    multi = dev_c[dev_c["customers"] >= 2]
    R["shared_infra"] = {
        "devices_used_by_2plus_customers": len(multi),
        "of_which_legit_only": int((multi["fraud"] == 0).sum()),
        "of_which_ring": int((multi["ring"] > 0).sum()),
        "of_which_other_fraud": int(((multi["fraud"] == 1) & (multi["ring"] == 0)).sum()),
        "legit_networks_used_by_2plus_customers": int((net_c >= 2).sum()),
        "production_ring_rule_precision_pct": _pct(multi["ring"] > 0) if len(multi) else float("nan"),
    }
    # fraud-only devices shared by customers outside one ring would be accidental collisions
    fraud_only = tx[fraud & ~tx["device_id"].isin(legit_devices)]
    for dev, g in fraud_only.groupby("device_id"):
        if g["customer_id"].nunique() > 1:
            check(g["fraud_ring_id"].nunique() == 1 and g["fraud_ring_id"].iloc[0] > 0,
                  f"fraud device {dev} shared across unrelated episodes")

    # ---- failed logins
    R["failed_logins"] = _failed_login_audit(tx, lf, check)

    # ---- metadata / leakage
    leak_feature = sorted(set(feature_cols) & forbidden)
    leak_dnn = sorted(set(dnn_cols) & forbidden)
    check(not leak_feature and not leak_dnn, "ground-truth columns in the production feature lists")
    check(list(feat.columns[: len(schema.V1_FEATURES_CSV_COLUMNS)]) == schema.V1_FEATURES_CSV_COLUMNS,
          "features file does not start with the v1 columns")
    check(list(tx.columns[:-1]) == schema.RAW_COLUMNS, "transactions columns differ from the schema")
    X, y, meta = fe.build_sequences(feat.drop(columns=["ts", "_n_signals"]))
    R["leakage"] = {
        "feature_columns": feature_cols, "dnn_input_columns": dnn_cols,
        "forbidden_in_feature_columns": leak_feature, "forbidden_in_dnn_columns": leak_dnn,
        "sequence_matrix_shape": list(X.shape), "sequence_fraud_targets": int(y.sum()),
        "features_file_columns": list(feat.columns[:-2]),
    }
    check(X.shape[1:] == (10, len(feature_cols)), "sequence matrix has an unexpected shape")

    # ---- giveaways
    kinds = pd.Series(np.select([tx["device_id"].str.match(ORIGINAL_DEVICE), tx["device_id"].str.match(NEUTRAL_DEVICE)],
                                ["DEV_nnnn_A/B", "DEV_XXXXXXXX"], "other"), index=tx.index)
    fmt = pd.crosstab(kinds, tx["is_fraud"]).rename(columns={0: "legit_rows", 1: "fraud_rows"})
    fmt["fraud_share_pct"] = (fmt.get("fraud_rows", 0) / fmt.sum(axis=1) * 100).round(2)
    city = tx.groupby("location").agg(rows=("is_fraud", "size"), fraud=("is_fraud", "sum"))
    city["legit"] = city["rows"] - city["fraud"]
    city["fraud_share_pct"] = (city["fraud"] / city["rows"] * 100).round(2)
    R["giveaways"] = {
        "dev_unknown_rows": int(tx["device_id"].str.contains("UNKNOWN").sum()),
        "device_formats": fmt.reset_index(names="format"),
        "cities": city.reset_index(),
        "fraud_only_cities": sorted(city.index[(city["fraud"] > 0) & (city["legit"] == 0)]),
    }
    check(R["giveaways"]["dev_unknown_rows"] == 0, "DEV_UNKNOWN_ device ids present")
    check((kinds != "other").all(), "device ids in an unexpected format")
    check(not R["giveaways"]["fraud_only_cities"], "cities that only ever appear on fraud")
    check((fmt.get("legit_rows", pd.Series(0, index=fmt.index)) > 0).all(), "a device-id format used only by fraud")

    # ---- distribution / integrity
    R["integrity"] = _integrity(tx, feat, cust, ep, lf, start, end, check)

    # ---- production feature compatibility
    R["compatibility"] = _compatibility(fe, tx, feat, feature_cols, recompute_features, check)
    return R


def _failed_login_audit(tx, lf, check):
    day = np.timedelta64(24, "h")
    by_c = {c: np.sort(g["ts"].to_numpy()) for c, g in lf.groupby("customer_id")}
    mismatches = 0
    classes = {"event exactly 24h before a transaction": 0, "event less than 1 minute inside the 24h window": 0,
               "event less than 1 minute outside the 24h window": 0, "event at the transaction's own timestamp": 0}
    for c, g in tx.groupby("customer_id"):
        et = by_c.get(c, np.array([], dtype="datetime64[ns]"))
        for ts, got in zip(g["ts"].to_numpy(), g["failed_logins_24h"].to_numpy()):
            if got != int(((et >= ts - day) & (et < ts)).sum()):
                mismatches += 1
            if len(et):
                d = ts - et                                    # positive = event before the transaction
                classes["event exactly 24h before a transaction"] += int((d == day).sum())
                classes["event less than 1 minute inside the 24h window"] += int(((d < day) & (d > day - np.timedelta64(60, "s"))).sum())
                classes["event less than 1 minute outside the 24h window"] += int(((d > day) & (d < day + np.timedelta64(60, "s"))).sum())
                classes["event at the transaction's own timestamp"] += int((d == np.timedelta64(0, "s")).sum())
    check(mismatches == 0, f"{mismatches} transactions have a failed_logins_24h that is not the preceding-24h count")

    # constructed boundary cases through the generator's own counting function
    t = 100 * DAY
    cases = [
        ("event exactly 24h before", [t - DAY], 1),
        ("event 1s inside 24h", [t - DAY + 1], 1),
        ("event 1s outside 24h", [t - DAY - 1], 0),
        ("event at the transaction timestamp", [t], 0),
        ("event 1s after the transaction", [t + 1], 0),
        ("all of the above", [t - DAY - 1, t - DAY, t - DAY + 1, t, t + 1], 2),
    ]
    results = []
    for name, events, expected in cases:
        got = int(count_preceding(np.array(sorted(events)), np.array([t]))[0])
        results.append({"case": name, "expected": expected, "counted": got, "ok": got == expected})
        check(got == expected, f"failed-login boundary case '{name}' counted {got}, expected {expected}")
    late = lf.merge(tx.groupby("customer_id")["ts"].max().rename("last"), on="customer_id", how="left")
    return {"transactions_checked": len(tx), "mismatches": mismatches, "real_near_boundary_events": classes,
            "constructed_cases": pd.DataFrame(results), "events": len(lf),
            "events_by_source": lf["source"].value_counts().to_dict(),
            "events_for_unknown_customers": int(late["last"].isna().sum())}


def _integrity(tx, feat, cust, ep, lf, start, end, check):
    out = {}
    num = tx.select_dtypes(include="number")
    out["nan_or_inf_numeric_values"] = int((~np.isfinite(num.to_numpy(dtype=float))).sum())
    fnum = feat.select_dtypes(include="number")
    out["nan_or_inf_feature_values"] = int((~np.isfinite(fnum.to_numpy(dtype=float))).sum())
    out["empty_string_values"] = int((tx.drop(columns=["ts"]) == "").sum().sum())
    out["amount_le_0"] = int((tx["amount"] <= 0).sum())
    out["amount_min"] = float(tx["amount"].min())
    out["amount_max"] = float(tx["amount"].max())
    out["amount_below_10"] = int((tx["amount"] < 10).sum())
    out["amount_below_10_non_probe_non_small"] = int(((tx["amount"] < 10) & ~(
        (tx["fraud_type"] == "card_testing_cashout") | tx["legit_context"].str.contains("small_purchase"))).sum())
    out["invalid_timestamps"] = int(tx["ts"].isna().sum())
    out["timestamps_outside_range"] = int(((tx["ts"] < start) | (tx["ts"] >= end)).sum())
    out["duplicate_transaction_ids"] = int(tx["transaction_id"].duplicated().sum())
    out["duplicate_transactions"] = int(tx.drop(columns=["transaction_id"]).duplicated().sum())
    out["duplicate_customer_timestamps"] = int(tx.duplicated(["customer_id", "timestamp"]).sum())
    out["customers_without_transactions"] = int(len(set(cust["customer_id"]) - set(tx["customer_id"])))
    out["transactions_for_unknown_customers"] = int((~tx["customer_id"].isin(cust["customer_id"])).sum())
    per_c = tx.groupby("customer_id").size()
    out["transactions_per_customer"] = {"min": int(per_c.min()), "median": int(per_c.median()), "max": int(per_c.max())}

    # device / customer relationships: an original DEV_nnnn_A/B device belongs to customer nnnn;
    # another customer may only use it as a borrowed device
    m = tx["device_id"].str.extract(ORIGINAL_DEVICE)[0]
    foreign_owner = m.notna() & (("CUST_" + m.fillna("")) != tx["customer_id"])
    out["original_device_used_by_other_customer"] = int(foreign_owner.sum())
    out["of_which_not_borrowed"] = int((foreign_owner & ~tx["legit_context"].str.contains("borrowed_device")).sum())
    out["devices_on_4plus_customers"] = int((tx.groupby("device_id")["customer_id"].nunique() >= 4).sum())
    out["devices_on_7plus_customers"] = int((tx.groupby("device_id")["customer_id"].nunique() >= 7).sum())
    out["login_events_outside_range"] = int(((lf["ts"] < start - pd.Timedelta(days=1)) | (lf["ts"] >= end) | lf["ts"].isna()).sum())
    out["login_events_unknown_customer"] = int((~lf["customer_id"].isin(cust["customer_id"])).sum())

    fr = tx[tx["is_fraud"] == 1]
    out["episodes_without_fraud_rows"] = int((~ep["fraud_episode_id"].isin(fr["fraud_episode_id"])).sum())
    out["fraud_rows_without_episode"] = int((~fr["fraud_episode_id"].isin(ep["fraud_episode_id"])).sum())
    firsts = fr.groupby("fraud_episode_id")["ts"].min()
    pre = tx[tx["is_precursor"] == 1]
    # each warning row must lie in [precursor_start, first fraud) of one of its customer's episodes,
    # i.e. strictly before the fraud it precedes
    inside = pd.Series(False, index=pre.index)
    windows = ep[ep["precursor_start"] != ""]
    for _, e in windows.iterrows():
        inside |= ((pre["customer_id"] == e["customer_id"]) & (pre["ts"] >= pd.Timestamp(e["precursor_start"]))
                   & (pre["ts"] < firsts.get(e["fraud_episode_id"], pd.Timestamp.min)))
    out["warning_rows_not_before_their_fraud"] = int((~inside).sum())
    out["warning_rows_labelled_fraud"] = int((pre["is_fraud"] == 1).sum())
    # legitimate rows inside a warning window that are not flagged
    missed = 0
    for _, e in windows.iterrows():
        m = ((tx["customer_id"] == e["customer_id"]) & (tx["is_fraud"] == 0)
             & (tx["ts"] >= pd.Timestamp(e["precursor_start"])) & (tx["ts"] < pd.Timestamp(e["first_fraud_time"])))
        missed += int((m & (tx["is_precursor"] == 0)).sum())
    out["warning_window_rows_not_flagged"] = missed
    out["ring_rows_outside_ring_window"] = 0
    for rid, g in tx[tx["fraud_ring_id"] > 0].groupby("fraud_ring_id"):
        e = ep[ep["fraud_ring_id"] == rid]
        lo, hi = pd.to_datetime(e["first_fraud_time"]).min(), pd.to_datetime(e["last_fraud_time"]).max()
        out["ring_rows_outside_ring_window"] += int(((g["ts"] < lo) | (g["ts"] > hi)).sum())

    # metadata consistency
    fraud = tx["is_fraud"] == 1
    meta_bad = {
        "is_fraud vs fraud_episode_id": int((fraud != (tx["fraud_episode_id"] > 0)).sum()),
        "is_fraud vs fraud_type": int((fraud != (tx["fraud_type"] != "none")).sum()),
        "is_fraud vs fraud_stage": int((fraud != tx["fraud_stage"].isin(["first", "subsequent"])).sum()),
        "ring id without ring type": int(((tx["fraud_ring_id"] > 0) != (tx["fraud_type"] == "ring")).sum()),
        "fraud row with legit_context": int((fraud & (tx["legit_context"] != "none")).sum()),
        "episodes with != 1 first stage": int((fr.groupby("fraud_episode_id")["fraud_stage"]
                                               .apply(lambda s: (s == "first").sum()) != 1).sum()),
        "first stage not the earliest": int((fr[fr["fraud_stage"] == "first"].set_index("fraud_episode_id")["ts"]
                                             .reindex(firsts.index) != firsts).sum()),
        "episode rows on more than one customer": int((fr.groupby("fraud_episode_id")["customer_id"].nunique() > 1).sum()),
        "episode table count mismatch": int((fr.groupby("fraud_episode_id").size().reindex(ep["fraud_episode_id"])
                                             .fillna(0).to_numpy() != ep["n_fraud_transactions"].to_numpy()).sum()),
        "segment differs from customers.csv": int((tx["customer_segment"] != tx["customer_id"].map(
            cust.set_index("customer_id")["customer_segment"])).sum()),
        "is_new_device mismatch": int((tx["is_new_device"] != (~tx.duplicated(["customer_id", "device_id"])).astype(int)).sum()),
        "is_new_location mismatch": int((tx["is_new_location"] != (~tx.duplicated(["customer_id", "location"])).astype(int)).sum()),
    }
    out["metadata_inconsistencies"] = meta_bad

    valid = {
        "merchant_category": set(MERCHANT_CATEGORIES), "location": set(DOMESTIC_CITIES) | set(FOREIGN_CITIES),
        "fraud_type": set(ARCHETYPES) | {"none"}, "fraud_stage": set(schema.FRAUD_STAGES),
        "customer_segment": set(cust["customer_segment"]), "is_fraud": {0, 1}, "is_precursor": {0, 1},
        "is_new_device": {0, 1}, "is_new_location": {0, 1},
    }
    malformed = {col: sorted(map(str, set(tx[col]) - allowed)) for col, allowed in valid.items() if set(tx[col]) - allowed}
    ctx_values = set(tx["legit_context"].str.split("|").explode()) - set(schema.LEGIT_CONTEXTS) - {"none"}
    if ctx_values:
        malformed["legit_context"] = sorted(ctx_values)
    if not tx["merchant_id"].str.match(MERCHANT).all():
        malformed["merchant_id"] = ["unexpected format"]
    if not tx["network_id"].str.match(NEUTRAL_NETWORK).all():
        malformed["network_id"] = ["unexpected format"]
    if not tx["transaction_id"].str.match(r"^TXN_\d{7}$").all():
        malformed["transaction_id"] = ["unexpected format"]
    out["malformed_categorical_values"] = malformed
    out["merchant_category_matches_merchant"] = int(tx.groupby("merchant_id")["merchant_category"].nunique().max() == 1)

    for key in ("nan_or_inf_numeric_values", "nan_or_inf_feature_values", "empty_string_values", "amount_le_0",
                "invalid_timestamps", "timestamps_outside_range", "duplicate_transaction_ids", "duplicate_transactions",
                "duplicate_customer_timestamps", "customers_without_transactions", "transactions_for_unknown_customers",
                "of_which_not_borrowed", "login_events_outside_range", "login_events_unknown_customer",
                "episodes_without_fraud_rows", "fraud_rows_without_episode", "warning_rows_not_before_their_fraud",
                "warning_rows_labelled_fraud", "warning_window_rows_not_flagged", "ring_rows_outside_ring_window"):
        check(out[key] == 0, f"integrity: {key} = {out[key]}")
    for key, n in meta_bad.items():
        check(n == 0, f"metadata: {key} = {n}")
    check(not malformed, f"malformed values: {malformed}")
    check(out["merchant_category_matches_merchant"] == 1, "a merchant appears under more than one category")
    return out


def _compatibility(fe, tx, feat, feature_cols, recompute, check):
    out = {"v1_input_columns_present": all(c in tx.columns for c in schema.V1_RAW_COLUMNS),
           "feature_columns": feature_cols, "feature_count": len(feature_cols),
           "matches_v1_feature_list": feature_cols == schema.V1_FEATURE_COLUMNS}
    check(out["v1_input_columns_present"], "v1 input columns missing")
    check(out["matches_v1_feature_list"], "production feature list changed")
    if recompute:
        raw = tx[schema.V1_RAW_COLUMNS].copy()
        again = fe.build_point_features(raw)
        again["timestamp"] = pd.to_datetime(again["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        a = again.set_index("transaction_id")[feature_cols]
        b = feat.set_index("transaction_id").loc[a.index, feature_cols]
        diff = float(np.abs(a.to_numpy(float) - b.to_numpy(float)).max())
        out["recomputed_rows"] = len(again)
        out["max_abs_difference_vs_features_file"] = diff
        out["recomputed_columns"] = [c for c in again.columns if c not in schema.V1_FEATURES_BASE_COLUMNS]
        check(diff < 1e-6, f"recomputed features differ from the features file (max {diff})")
        check(out["recomputed_columns"] == feature_cols, "feature engineering produced unexpected columns")
    return out


# ---- markdown ------------------------------------------------------------------------

def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.to_dict("records"):            # keeps each column's own type (no int -> float upcast)
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) < 1e6 else f"{v:,.0f}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return str(v)


def _kv(d: dict) -> str:
    return _table(pd.DataFrame({"item": list(d), "value": [_fmt(v) if not isinstance(v, (dict, list)) else
                                                           json.dumps(v) for v in d.values()]}))


def render(R: dict, manifest: dict) -> str:
    s = R["structure"]
    L = []
    add = L.append
    add("# v2 synthetic data — sanity report\n")
    add("Generated by `python data/v2/sanity_report.py` from the files in `data/v2/`. Every number is recomputed "
        "from the CSV files. The report has no wall-clock timestamp: the same data gives the same report.\n")
    verdict = "**No anomalies found.**" if not R["anomalies"] else \
        "**ANOMALIES FOUND:**\n\n" + "\n".join(f"- {a}" for a in R["anomalies"])
    add(f"## Verdict\n\n{verdict}\n")
    add("## 1. Files and reproducibility\n")
    add(f"Generator `{manifest['generator']}` version {manifest['generator_version']}, seed {manifest['seed']}, "
        f"command `{manifest['command']}`.\n")
    add(_table(pd.DataFrame([{"file": k, "rows": v["rows"], "bytes": v["bytes"], "sha256": v["sha256"]}
                             for k, v in R["files"].items()])) + "\n")
    add("## 2. Dataset structure\n")
    add(_kv({
        "transactions": s["transactions"], "customers": s["customers"],
        "configured date range": s["configured_range"],
        "first / last transaction": f"{s['first_timestamp']} / {s['last_timestamp']}",
        "fraud transactions": f"{s['fraud']:,} ({s['fraud_pct']}%)",
        "legitimate transactions": f"{s['legit']:,} ({s['legit_pct']}%)",
        "fraud episodes": s["episodes"], "ring episodes": s["ring_episodes"], "rings": s["rings"],
        "ring sizes (victims: rings)": ", ".join(f"{k}: {v}" for k, v in s["ring_sizes"].items()),
        "episodes with a warning period": s["episodes_with_warning_period"],
        "warning-period transactions (legitimate)": s["warning_period_transactions"],
        "households / customers in households": f"{s['households']} / {s['customers_in_households']}",
    }) + "\n")
    add("Episodes by fraud type:\n")
    add(_table(pd.DataFrame(list(s["episodes_by_type"].items()), columns=["fraud type", "episodes"])) + "\n")

    add("## 3. Class balance\n")
    add("Reported as generated; nothing was tuned toward a target. How to read it: every customer has the same "
        "chance of becoming a victim, so the per-transaction fraud rate is higher for customer types that transact "
        "less often. No episode starts before day `first_episode_day` of the configured range (history warm-up), so "
        "the first weeks have no fraud; the last month is a partial month.\n")
    add("By customer type:\n\n" + _table(R["balance"]["by_segment"]) + "\n")
    add("By fraud type:\n\n" + _table(R["balance"]["by_type"]) + "\n")
    add("By month:\n\n" + _table(R["balance"]["by_month"]) + "\n")

    add("## 4. Warning-sign overlap (production features)\n")
    add("Prevalence of each signal, computed from the production feature columns in `transactions_with_features.csv`. "
        "`legit warning period` = legitimate transactions inside an episode's warning period.\n")
    add(_table(R["overlap"]["signals"].rename(columns={
        "fraud_pct": "fraud %", "legit_pct": "legit %", "legit_warning_period_pct": "legit warning period %",
        "legit_outside_warning_pct": "legit outside warning %"})) + "\n")
    fc, lc = R["overlap"]["fraud_signal_count"], R["overlap"]["legit_signal_count"]
    tot_f, tot_l = sum(fc.values()), sum(lc.values())
    add("Number of the 7 main signals present per row:\n")
    add(_table(pd.DataFrame([{"signals": k, "fraud rows": fc[k], "fraud %": round(fc[k] / tot_f * 100, 1),
                              "legit rows": lc[k], "legit %": round(lc[k] / tot_l * 100, 2)} for k in fc])) + "\n")
    sep = R["overlap"]["single_threshold_separable"]
    add(f"Features that separate fraud from legitimate rows with a single threshold: "
        f"{'none' if not sep else ', '.join(sep)}.\n")
    v = R["v1_rule"]
    add(f"**Old v1 rule** (`hour_is_unusual AND amount_pct_of_avg >= 188`, threshold unchanged): flags {v['flagged']:,} rows, "
        f"{v['true_positives']} of them fraud — **precision {v['precision']:.3f}, recall {v['recall']:.3f}** "
        f"(on v1 both were 1.000).\n")

    add("## 5. Legitimate unusual behaviour\n")
    add("All rows below are `is_fraud = 0` (column `fraud_rows` must be 0). A row can carry several contexts.\n")
    add(_table(R["legit_unusual"].rename(columns={"context": "legit_context"})) + "\n")
    add(_kv({"legitimate rows in foreign cities": R["legit_extra"]["foreign_rows"],
             "customers with legitimate foreign rows": R["legit_extra"]["foreign_customers"],
             "legitimate rows with failed logins in the previous 24h": R["legit_extra"]["rows_with_failed_logins"],
             "customers with a device upgrade": R["legit_extra"]["customers_with_upgrade"],
             "customers with a secondary device": R["legit_extra"]["customers_with_secondary_device"]}) + "\n")

    add("## 6. Fraud archetypes\n")
    add("Signal columns are % of that archetype's fraud rows. `zero_signals_pct` = fraud rows with none of the 7 signals.\n")
    add(_table(R["archetypes"]) + "\n")
    add("Signature check: combinations of (own original device vs other device) x (domestic vs foreign) x category "
        "per archetype. A single combination would be a deterministic signature.\n")
    add(_table(R["signatures"]) + "\n")

    add("## 7. Fraud rings\n")
    add("Shared infrastructure = a device, network or merchant used by two or more members of the same ring.\n")
    add(_table(R["rings"]) + "\n")
    add(_kv({k.replace("_", " "): v for k, v in R["shared_infra"].items()}) + "\n")
    add("`other fraud` = a device used by several customers where one of them was also defrauded on it, e.g. a "
        "customer's own phone that another customer borrowed legitimately and on which a fraud against its owner "
        "also happened (a fraud type that uses the victim's own device). The audit separately checks that no fraudster device is shared across unrelated "
        "episodes. `production ring rule precision` = share of multi-customer devices that belong to a ring: "
        "the production ring detector (any device used by 2+ customers) would mostly flag legitimate sharing.\n")

    add("## 8. Failed-login correctness\n")
    fl = R["failed_logins"]
    add(f"`failed_logins_24h` was recomputed independently for all {fl['transactions_checked']:,} transactions from "
        f"`login_failures.csv` as the number of events in `[t - 24h, t)`: **{fl['mismatches']} mismatches**.\n")
    add("Constructed boundary cases (generator's counting function):\n\n" + _table(fl["constructed_cases"]) + "\n")
    add("Near-boundary events that occur in the real data (all included in the recomputation above):\n")
    add(_kv(fl["real_near_boundary_events"]) + "\n")
    add(_kv({"login-failure events": fl["events"], "events by source": fl["events_by_source"],
             "events for unknown customers": fl["events_for_unknown_customers"]}) + "\n")

    add("## 9. Metadata / leakage audit\n")
    lk = R["leakage"]
    add(_kv({
        "production FEATURE_COLUMNS": lk["feature_columns"],
        "DNN_INPUT_COLUMNS (read from dnn_model.py)": lk["dnn_input_columns"],
        "ground-truth columns in FEATURE_COLUMNS": lk["forbidden_in_feature_columns"] or "none",
        "ground-truth columns in DNN_INPUT_COLUMNS": lk["forbidden_in_dnn_columns"] or "none",
        "LSTM sequence matrix from the features file": f"{lk['sequence_matrix_shape']} (only FEATURE_COLUMNS)",
    }) + "\n")
    add("The features file itself has the v1 columns, then the metadata columns appended at the end (by design, "
        "for analysis). The model matrices are built only from `FEATURE_COLUMNS` (+ `risk_score` for the DNN) and "
        "`is_fraud` as the target, so the metadata never reaches a model. Columns of the features file:\n")
    add("`" + "`, `".join(lk["features_file_columns"]) + "`\n")

    add("## 10. Giveaway checks\n")
    g = R["giveaways"]
    add(f"- Device ids containing `UNKNOWN`: **{g['dev_unknown_rows']}**.\n"
        f"- Cities that appear only on fraud rows: **{', '.join(g['fraud_only_cities']) or 'none'}**.\n")
    add("Device-id formats:\n\n" + _table(g["device_formats"]) + "\n")
    add("Locations:\n\n" + _table(g["cities"].sort_values("fraud_share_pct", ascending=False)) + "\n")

    add("## 11. Integrity checks\n")
    integ = {k: v for k, v in R["integrity"].items() if k not in ("metadata_inconsistencies", "malformed_categorical_values")}
    add(_kv(integ) + "\n")
    add("Metadata consistency (all must be 0):\n\n" + _kv(R["integrity"]["metadata_inconsistencies"]) + "\n")
    mal = R["integrity"]["malformed_categorical_values"]
    add(f"Malformed categorical values: {'none' if not mal else json.dumps(mal)}.\n")

    add("## 12. Production feature compatibility\n")
    add(_kv({k.replace("_", " "): v for k, v in R["compatibility"].items()}) + "\n")
    return "\n".join(L)
