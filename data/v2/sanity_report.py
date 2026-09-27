"""
Audit a generated v2 dataset and write DATA_SANITY_REPORT.md.

    python data/v2/sanity_report.py                   # audits data/v2
    python data/v2/sanity_report.py --dir /tmp/sample --out /tmp/sample/report.md

Exits with status 1 if any anomaly is found.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from synth_v2.sanity import audit, render  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=str(HERE), help="dataset directory (default: data/v2)")
    p.add_argument("--out", default=None, help="report path (default: <dir>/DATA_SANITY_REPORT.md)")
    p.add_argument("--skip-feature-recompute", action="store_true",
                   help="do not re-run production feature engineering (faster)")
    args = p.parse_args(argv)
    directory = Path(args.dir)
    results = audit(directory, recompute_features=not args.skip_feature_recompute)
    manifest = json.loads((directory / "manifest.json").read_text())
    out = Path(args.out) if args.out else directory / "DATA_SANITY_REPORT.md"
    out.write_text(render(results, manifest), encoding="utf-8")
    print(f"wrote {out}")
    if results["anomalies"]:
        print("ANOMALIES:\n  " + "\n  ".join(results["anomalies"]))
        return 1
    print("no anomalies found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
