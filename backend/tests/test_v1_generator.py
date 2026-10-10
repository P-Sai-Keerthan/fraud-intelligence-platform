"""The v1 seed-data generator must be reproducible (audit finding R-09) and the timestamp window enforced (R-06)."""

import importlib.util
import hashlib
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_generator():
    spec = importlib.util.spec_from_file_location("v1_generator", ROOT / "data" / "generate_synthetic_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(df):
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def test_same_seed_and_start_date_give_identical_data(monkeypatch):
    gen = _load_generator()
    monkeypatch.setattr(gen, "N_CUSTOMERS", 30)
    first, second = gen.generate_dataset(), gen.generate_dataset()
    assert _digest(first) == _digest(second)
    assert first["timestamp"].min() >= datetime(2026, 1, 12)          # anchored to the fixed date, not to the clock


def test_a_different_seed_changes_the_data(monkeypatch):
    gen = _load_generator()
    monkeypatch.setattr(gen, "N_CUSTOMERS", 30)
    assert _digest(gen.generate_dataset()) != _digest(gen.generate_dataset(seed=7))


@pytest.mark.parametrize("stamp, ok", [("1999-12-31T23:59:59", False), ("2000-01-01T00:00:00", True), ("2026-07-10T13:30:00", True),
                                      ("2099-12-31T23:59:59", True), ("2100-01-01T00:00:00", False), ("0001-01-01T00:00:00", False),
                                      ("9999-12-31T23:59:59", False)])
def test_timestamp_window(client, normal_txn, stamp, ok):
    r = client.post("/predict", json=dict(normal_txn, timestamp=stamp))
    assert (r.status_code == 200) == ok, r.text
    if not ok:
        assert r.status_code == 422 and "timestamp must be between" in r.text
