"""
Multi-seed candidate training (Step 4C-3E, second evidence stage).

    cd backend
    python -m app.training.multiseed                 # every seed in TRAINING_SEEDS that is not trained yet
    python -m app.training.multiseed --seed 11       # one seed

Purpose: measure how much each candidate's results move between training
runs, so the v2_dnn_lstm / v2_dnn_only comparison is not read off a single
run (seed 42).

What changes between runs: the training seed only. It seeds the weight
initialisation, the shuffling, dropout and the customer-grouped folds of the
out-of-fold LSTM scores. Everything else is the existing candidate recipe,
called through the same function (app.training.candidates.train_candidates):
the same seed-42 v2 dataset and saved time split, the same features, scaling,
clipping, architectures, hyperparameters, early stopping and validation rows.
Nothing is tuned.

Artifacts go to models/candidates_multiseed/v2/seed_<seed>/{dnn_lstm,dnn_only}/,
a location separate from production (models/saved/) and from the seed-42
candidates (models/candidates/). This module

* refuses to write inside models/saved/ or models/candidates/;
* refuses to train a seed whose directory already holds a manifest (no overwrite);
* compares the SHA-256 of every production and seed-42 candidate file before
  and after, and raises if any changed.

It does not select, promote or deploy anything. MODEL_SET is not involved:
the application cannot load these directories by a model-set name.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .. import config
from . import candidates as cand

# Fixed before any of these runs was trained. Seed 42 is the existing candidates'
# seed and is deliberately not among them; data seeds 101-105 / 201-205 are not reused.
TRAINING_SEEDS = (11, 12, 13, 14, 15)
REFERENCE_SEED = 42
MULTISEED_DIR = config.BACKEND_DIR / "models" / "candidates_multiseed"
DATASET = cand.REQUIRED_DATASET
RUN_FILE = "training_run.json"


class MultiSeedError(ValueError):
    """An unsafe output location or a seed that must not be trained here."""


def seed_root(seed: int, root=None) -> Path:
    """Directory holding dnn_lstm/ and dnn_only/ of one training seed."""
    return Path(root or MULTISEED_DIR) / DATASET / f"seed_{int(seed)}"


def assert_safe_root(path) -> Path:
    p = Path(path).resolve()
    for protected in (config.MODELS_SAVED_DIR.resolve(), config.CANDIDATES_DIR.resolve()):
        if p == protected or protected in p.parents:
            raise MultiSeedError(f"refusing to write multi-seed candidates into {protected}")
    return p


def protected_checksums() -> dict:
    """SHA-256 of every file under models/saved/ and models/candidates/."""
    out = {}
    for label, root in (("production", config.MODELS_SAVED_DIR), ("candidates", config.CANDIDATES_DIR)):
        out[label] = {f.relative_to(root).as_posix(): cand._sha256(f) for f in sorted(root.rglob("*")) if f.is_file()} \
            if root.exists() else {}
    return out


def train_seed(seed: int, root=None, spec=None, log=print, **train_kwargs) -> dict:
    """Trains both candidates with one training seed and writes them under
    seed_root(seed, root). Returns the run record (also saved as training_run.json)."""
    seed = int(seed)
    if seed == REFERENCE_SEED and root is None:
        raise MultiSeedError(f"seed {REFERENCE_SEED} is the existing candidates' seed; it is not retrained here")
    out = assert_safe_root(seed_root(seed, root))
    for name in cand.CANDIDATES:
        if (out / name / cand.MANIFEST).exists():
            raise MultiSeedError(f"{out / name} already holds a trained candidate; it is never overwritten")
    before = protected_checksums()
    started = time.time()
    result = cand.train_candidates(DATASET, names=cand.CANDIDATES, output_root=out, spec=spec, seed=seed, log=log,
                                   **train_kwargs)
    seconds = round(time.time() - started, 1)
    after = protected_checksums()
    if after != before:
        raise RuntimeError("production or seed-42 candidate files changed during multi-seed training")
    manifests = result["manifests"]
    record = {
        "training_seed": seed,
        "candidates": sorted(manifests),
        "dataset": manifests["dnn_only"]["dataset"],
        "split": manifests["dnn_only"]["split"],
        "features": manifests["dnn_only"]["features"],
        "recipe": manifests["dnn_only"]["recipe"],
        "weights_sha256": {
            "dnn_lstm": {"dnn": manifests["dnn_lstm"]["model"]["dnn_weights_sha256"],
                         "lstm": manifests["dnn_lstm"]["model"]["lstm_weights_sha256"]},
            "dnn_only": {"dnn": manifests["dnn_only"]["model"]["dnn_weights_sha256"]},
        },
        "files_sha256": {n: m["files"] for n, m in manifests.items()},
        "scalers_sha256": {n: m["scalers"]["sha256"] for n, m in manifests.items()},
        "epochs_run": {
            "dnn_lstm": {"dnn": manifests["dnn_lstm"]["model"]["dnn_training"].get("epochs_run"),
                         "lstm": manifests["dnn_lstm"]["model"]["lstm_training"].get("epochs_run")},
            "dnn_only": {"dnn": manifests["dnn_only"]["model"]["dnn_training"].get("epochs_run")},
        },
        "training_seconds": seconds,
        "versions": manifests["dnn_only"]["versions"],
        "production_and_seed_42_candidate_files_unchanged": True,
        "protected_files_sha256": before,
    }
    (out / RUN_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return record


def trained_seeds(root=None) -> list:
    return [s for s in TRAINING_SEEDS
            if all((seed_root(s, root) / n / cand.MANIFEST).exists() for n in cand.CANDIDATES)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=None, help=f"train one seed (default: all of {TRAINING_SEEDS})")
    p.add_argument("--root", default=None, help=f"default: {MULTISEED_DIR}")
    args = p.parse_args(argv)
    if args.seed is not None:
        record = train_seed(args.seed, args.root)
        print(json.dumps({k: record[k] for k in ("training_seed", "training_seconds", "epochs_run")}))
        return record
    # one process per seed, so every run starts from a fresh TensorFlow state
    done = trained_seeds(args.root)
    for seed in TRAINING_SEEDS:
        if seed in done:
            print(f"seed {seed}: already trained, kept")
            continue
        command = [sys.executable, "-m", "app.training.multiseed", "--seed", str(seed)]
        if args.root:
            command += ["--root", args.root]
        subprocess.run(command, check=True, cwd=str(config.BACKEND_DIR))
    return trained_seeds(args.root)


if __name__ == "__main__":
    main()
