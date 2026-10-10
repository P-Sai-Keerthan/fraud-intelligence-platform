"""THROUGHPUT BENCHMARK ONLY. Random tensors, random labels: measures speed/memory of GRU configs, produces NO result about fraud.
usage: bench_dl.py torch|tf"""
import sys, time, json, resource, os
fw = sys.argv[1]; THREADS = 3
K, D, CTX, BATCH = 32, 26, 45, 512
CONFIGS = [(32, 1), (64, 1), (128, 1), (64, 2), (128, 2)]
out = {"framework": fw, "threads": THREADS, "K": K, "step_dim": D, "context_dim": CTX, "batch": BATCH, "rows": []}
import numpy as np
rng = np.random.default_rng(0)
X = rng.standard_normal((BATCH * 24, K, D), dtype=np.float32); C = rng.standard_normal((BATCH * 24, CTX), dtype=np.float32); y = (rng.random(BATCH * 24) < 0.05).astype(np.float32)
if fw == "torch":
    import torch, torch.nn as nn
    torch.set_num_threads(THREADS); out["version"] = torch.__version__
    class Net(nn.Module):
        def __init__(s, h, l):
            super().__init__(); s.gru = nn.GRU(D, h, l, batch_first=True); s.head = nn.Sequential(nn.Linear(h + CTX, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(s, x, c): o, hn = s.gru(x); return s.head(torch.cat([hn[-1], c], 1)).squeeze(1)
    Xt, Ct, yt = map(torch.from_numpy, (X, C, y))
    for h, l in CONFIGS:
        m = Net(h, l); opt = torch.optim.Adam(m.parameters(), 1e-3); lossf = nn.BCEWithLogitsLoss()
        npar = sum(p.numel() for p in m.parameters())
        def step(i):
            sl = slice(i * BATCH, (i + 1) * BATCH); opt.zero_grad(); loss = lossf(m(Xt[sl], Ct[sl]), yt[sl]); loss.backward(); opt.step()
        for i in range(3): step(i)
        t = time.time(); n = 18
        for i in range(n): step(i % 24)
        train_sps = n * BATCH / (time.time() - t)
        m.eval()
        with torch.no_grad():
            t = time.time(); [m(Xt[i*256:(i+1)*256], Ct[i*256:(i+1)*256]) for i in range(40)]; inf_sps = 40 * 256 / (time.time() - t)
            t = time.time(); [m(Xt[i:i+1], Ct[i:i+1]) for i in range(200)]; lat1 = (time.time() - t) / 200 * 1000
        out["rows"].append(dict(hidden=h, layers=l, params=npar, train_samples_per_s=round(train_sps), infer_samples_per_s_batch256=round(inf_sps), latency_ms_batch1=round(lat1, 2)))
else:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(THREADS); tf.config.threading.set_inter_op_parallelism_threads(1); out["version"] = tf.__version__
    from tensorflow import keras
    from tensorflow.keras import layers
    for h, l in CONFIGS:
        xi = keras.Input((K, D)); ci = keras.Input((CTX,)); z = xi
        for j in range(l): z = layers.GRU(h, return_sequences=(j < l - 1))(z)
        z = layers.Concatenate()([z, ci]); z = layers.Dense(64, activation="relu")(z); o = layers.Dense(1, activation="sigmoid")(z)
        m = keras.Model([xi, ci], o); m.compile("adam", "binary_crossentropy")
        for i in range(3): m.train_on_batch([X[:BATCH], C[:BATCH]], y[:BATCH])
        t = time.time(); n = 18
        for i in range(n): sl = slice((i % 24) * BATCH, (i % 24 + 1) * BATCH); m.train_on_batch([X[sl], C[sl]], y[sl])
        train_sps = n * BATCH / (time.time() - t)
        t = time.time(); [m.predict_on_batch([X[i*256:(i+1)*256], C[i*256:(i+1)*256]]) for i in range(40)]; inf_sps = 40 * 256 / (time.time() - t)
        t = time.time(); [m.predict_on_batch([X[i:i+1], C[i:i+1]]) for i in range(100)]; lat1 = (time.time() - t) / 100 * 1000
        out["rows"].append(dict(hidden=h, layers=l, params=int(m.count_params()), train_samples_per_s=round(train_sps), infer_samples_per_s_batch256=round(inf_sps), latency_ms_batch1=round(lat1, 2)))
out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
print(json.dumps(out, indent=1))
