"""THROUGHPUT BENCHMARK ONLY. Random features with a weak planted signal at 0.46 % positives; produces NO result about fraud."""
import time, json, resource, sys
import numpy as np
THREADS = 3
rng = np.random.default_rng(0)
def make(n, d):
    y = (rng.random(n) < 0.0046).astype(np.int8)
    X = rng.standard_normal((n, d), dtype=np.float32)
    X[y == 1, :5] += 1.5            # weak planted signal so trees are not degenerate
    return X, y
out = []
from sklearn.ensemble import RandomForestClassifier
for n in (100_000, 200_000, 400_000):
    X, y = make(n, 45)
    m = RandomForestClassifier(n_estimators=60, min_samples_leaf=5, max_features="sqrt", n_jobs=THREADS, random_state=0)
    t = time.time(); m.fit(X, y); fit = time.time() - t
    t = time.time(); m.predict_proba(X[:100_000]); pred = time.time() - t
    out.append(dict(model="RF(60 trees, leaf=5, sqrt)", rows=n, feats=45, fit_s=round(fit, 1), s_per_tree=round(fit / 60, 2), predict_100k_s=round(pred, 2)))
    print(json.dumps(out[-1]), flush=True)
import lightgbm as lgb
for n, d in ((200_000, 45), (200_000, 877)):
    X, y = make(n, d)
    ds = lgb.Dataset(X, y, params={"max_bin": 63}, free_raw_data=False)
    t = time.time(); b = lgb.train({"objective": "binary", "num_leaves": 31, "learning_rate": 0.1, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                                    "min_child_samples": 20, "num_threads": THREADS, "verbose": -1, "max_bin": 63}, ds, num_boost_round=100); fit = time.time() - t
    t = time.time(); b.predict(X[:100_000]); pred = time.time() - t
    out.append(dict(model="LightGBM(100 rounds, 31 leaves)", rows=n, feats=d, fit_s=round(fit, 1), s_per_round=round(fit / 100, 3), predict_100k_s=round(pred, 2)))
    print(json.dumps(out[-1]), flush=True)
print(json.dumps({"peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}))
