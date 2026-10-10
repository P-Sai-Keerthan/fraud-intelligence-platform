"""Regenerate the committed seed-42 dataset and compare SHA-256 with data/v2/manifest.json. usage: verify_determinism.py <manifest.json> <regenerated_dir>"""
import hashlib, json, sys
from pathlib import Path
man = json.load(open(sys.argv[1]))["files"]; d = Path(sys.argv[2])
ok = True
for f, meta in man.items():
    h = hashlib.sha256((d / f).read_bytes()).hexdigest(); ok &= h == meta["sha256"]
    print(f"{f:34s} matches committed manifest: {h == meta['sha256']}")
print("ALL MATCH" if ok else "MISMATCH"); sys.exit(0 if ok else 1)
