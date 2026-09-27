"""
Generate the v2 synthetic dataset.

    python data/v2/generate.py                    # default settings -> data/v2/
    python data/v2/generate.py --customers 60 --out /tmp/sample --skip-features

Writes transactions.csv, transactions_with_features.csv, customers.csv,
episodes.csv, login_failures.csv and manifest.json. The CSVs are not
committed to git (data/v2/.gitignore); the manifest records their checksums
so a regeneration can be verified. The application keeps using the v1 data
until it is explicitly switched.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from synth_v2 import DEFAULT_CONFIG, generate  # noqa: E402
from synth_v2.features import build_features_table  # noqa: E402
from synth_v2.output import write_dataset  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(HERE), help="output directory (default: data/v2)")
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    p.add_argument("--customers", type=int, default=DEFAULT_CONFIG.n_customers)
    p.add_argument("--skip-features", action="store_true", help="do not build transactions_with_features.csv")
    args = p.parse_args(argv)

    cfg = DEFAULT_CONFIG.with_overrides(seed=args.seed, n_customers=args.customers)
    data = generate(cfg)
    features = None if args.skip_features else build_features_table(data.transactions)
    # the recorded command lists only the options that change the content (not --out),
    # so regenerating into another directory gives a byte-identical manifest
    command = f"python data/v2/generate.py --seed {args.seed} --customers {args.customers}"
    if args.skip_features:
        command += " --skip-features"
    manifest = write_dataset(data, args.out, features, command=command)
    s = manifest["summary"]
    print(f"{s['transactions']:,} transactions, {s['customers']} customers, "
          f"{s['fraud_transactions']} fraud ({s['fraud_rate_pct']}%), {s['episodes']} episodes, {s['rings']} rings")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
