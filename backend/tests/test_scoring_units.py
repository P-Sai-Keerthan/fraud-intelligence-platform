"""Unit tests for the pure scoring helpers (no API, no model loading)."""

import numpy as np
import pytest

from app.models.dnn_model import alert_level_from_probability
from app.models.lstm_model import risk_probability_to_score
from app.models.similarity import compute_similarity


@pytest.mark.parametrize(
    "prob, level",
    [
        (0.0, "Low Risk"),
        (0.2499, "Low Risk"),
        (0.25, "Medium Risk"),
        (0.4999, "Medium Risk"),
        (0.50, "High Risk"),
        (0.7999, "High Risk"),
        (0.80, "Critical Risk"),
        (0.999, "Critical Risk"),
    ],
)
def test_alert_level_thresholds(prob, level):
    assert alert_level_from_probability(prob) == level


@pytest.mark.parametrize("prob, score", [(0.0, 0.0), (0.12345, 12.35), (0.5, 50.0), (1.0, 100.0)])
def test_risk_probability_to_score(prob, score):
    assert risk_probability_to_score(prob) == score


def test_similarity_identical_is_100():
    mean = np.array([1.0, 0.0, 5.0])
    std = np.array([0.5, 0.1, 2.0])
    result = compute_similarity(mean, mean, std)
    assert result["similarity_pct"] == 100.0
    assert result["deviation_pct"] == 0.0


def test_similarity_decreases_with_deviation():
    mean = np.zeros(9)
    std = np.ones(9)
    near = compute_similarity(np.full(9, 0.5), mean, std)["similarity_pct"]
    far = compute_similarity(np.full(9, 5.0), mean, std)["similarity_pct"]
    assert 0 <= far < near < 100


def test_similarity_handles_zero_std():
    result = compute_similarity(np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    assert np.isfinite(result["similarity_pct"])
    assert 0 <= result["similarity_pct"] <= 100
    assert result["similarity_pct"] + result["deviation_pct"] == pytest.approx(100)
