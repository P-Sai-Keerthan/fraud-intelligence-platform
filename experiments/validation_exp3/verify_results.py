"""
Independent recomputation of the Experiment 3 headline numbers from the per-row score files, using scikit-learn
instead of the experiment's own weighted-rank code. Exits non-zero if any value differs from results/*.json.

    python experiments/validation_exp3/verify_results.py --scores <scores_dir>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
TEST_SEEDS = (9303, 9304, 9305)
DEV_SEEDS = (9301, 9302)
EPS = 1e-6


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    args = ap.parse_args(argv)
    load = lambda seeds: pd.concat([pd.read_csv(Path(args.scores) / f"scores_seed_{s}.csv.gz") for s in seeds], ignore_index=True)  # noqa: E731
    test, dev = load(TEST_SEEDS), load(DEV_SEEDS)
    abl = json.loads((HERE / "results/exp3_ablation.json").read_text())
    sim = json.loads((HERE / "results/exp3_similarity.json").read_text())
    cal = json.loads((HERE / "results/exp3_calibration.json").read_text())
    checks, failed = [], []

    def check(name, got, expected, tol=2e-6):
        ok = abs(got - expected) <= tol
        checks.append((name, got, expected, ok))
        if not ok:
            failed.append(name)

    p10t, p10d = test["n_prior"] >= 10, dev["n_prior"] >= 10
    y = test["is_fraud"].to_numpy()
    # counts
    check("test rows", len(test), abl["meta"]["populations"]["test"]["rows"], 0)
    check("test fraud", int(y.sum()), abl["meta"]["populations"]["test"]["fraud"], 0)
    # ranking metrics (P10)
    for k in ("default", "dnn_lstm", "dnn_only", "production", "rf_nolstm", "logreg", "rule"):
        s = test.loc[p10t, k].to_numpy()
        r = abl["pooled_test"]["p10"][k]
        check(f"{k} ROC-AUC p10", roc_auc_score(y[p10t], s), r["roc_auc"])
        check(f"{k} PR-AUC p10", average_precision_score(y[p10t], s), r["pr_auc"])
        # alert metrics at the stored cut-off
        cut = abl["cutoffs"][k]
        a = test.loc[p10t, k].to_numpy() >= cut
        yt = y[p10t].astype(bool)
        tp, fp = int((a & yt).sum()), int((a & ~yt).sum())
        check(f"{k} TP", tp, r["confusion"]["tp"], 0)
        check(f"{k} FP", fp, r["confusion"]["fp"], 0)
        check(f"{k} recall", tp / yt.sum(), r["recall"])
        # the cut-off must meet the validation false-positive rate target
        sd = dev.loc[p10d, k].to_numpy()
        neg = dev.loc[p10d, "is_fraud"].to_numpy() == 0
        check(f"{k} validation FPR <= 1%", float((sd[neg] >= cut).mean() <= 0.01), 1.0, 0)
    # similarity
    d = test.loc[p10t, "dev_s0"].to_numpy()
    check("similarity ROC-AUC p10", roc_auc_score(y[p10t], d), sim["discrimination_pooled_test"]["s0_p10"]["roc_auc"])
    check("similarity PR-AUC p10", average_precision_score(y[p10t], d), sim["discrimination_pooled_test"]["s0_p10"]["pr_auc"])
    # calibration (all rows, probability clipped as in the experiment)
    p = np.clip(test["default"].to_numpy(), EPS, 1 - EPS)
    check("Brier uncalibrated", brier_score_loss(y, p), cal["pooled_test_all_rows"]["uncalibrated"]["brier"], 1e-8)
    check("log loss uncalibrated", log_loss(y, p), cal["pooled_test_all_rows"]["uncalibrated"]["log_loss"], 1e-6)
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(dev["default"].to_numpy(), dev["is_fraud"].to_numpy())
    pi = np.clip(iso.predict(test["default"].to_numpy()), EPS, 1 - EPS)
    check("Brier isotonic", brier_score_loss(y, pi), cal["pooled_test_all_rows"]["isotonic"]["brier"], 1e-8)
    check("log loss isotonic", log_loss(y, pi), cal["pooled_test_all_rows"]["isotonic"]["log_loss"], 1e-6)
    # protocol facts
    check("seeds are the five declared seeds", float(sorted(test["seed"].unique()) == list(TEST_SEEDS)
                                                    and sorted(dev["seed"].unique()) == list(DEV_SEEDS)), 1.0, 0)
    for name, got, exp, ok in checks:
        print(f"{'OK  ' if ok else 'FAIL'} {name}: recomputed {got:.8g} vs stored {exp:.8g}")
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
