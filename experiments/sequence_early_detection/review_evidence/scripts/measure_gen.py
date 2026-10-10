"""Infrastructure timing only (scratch seed 9). Times generation and feature building separately, records peak RSS and disk. Never used for results."""
import resource, sys, time, json
from pathlib import Path
SRC = Path(sys.argv[1]); OUT = Path(sys.argv[2]); N = int(sys.argv[3]); SEED = int(sys.argv[4])
sys.path.insert(0, str(SRC / "data" / "v2"))
from synth_v2 import DEFAULT_CONFIG, generate
from synth_v2.features import build_features_table
from synth_v2.output import write_dataset
t0 = time.time()
cfg = DEFAULT_CONFIG.with_overrides(seed=SEED, n_customers=N)
data = generate(cfg); t1 = time.time()
feat = build_features_table(data.transactions); t2 = time.time()
man = write_dataset(data, str(OUT), feat, command=f"scratch timing seed {SEED} customers {N}"); t3 = time.time()
s = man["summary"]
rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
disk = {p.name: p.stat().st_size for p in OUT.iterdir()}
print(json.dumps({"customers": N, "seed": SEED, "generate_s": round(t1-t0,1), "features_s": round(t2-t1,1), "write_s": round(t3-t2,1),
                  "peak_rss_mb": round(rss_mb), "transactions": s["transactions"], "fraud_transactions": s["fraud_transactions"],
                  "episodes": s["episodes"], "episodes_with_precursor": s["episodes_with_precursor"],
                  "precursor_transactions": s["precursor_transactions"], "rings": s["rings"],
                  "disk_bytes": disk, "disk_total_mb": round(sum(disk.values())/1048576,1)}, indent=1))
