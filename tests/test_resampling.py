"""Tests for resampling / imbalance handling."""
import numpy as np
import pandas as pd

from src import resampling


def _imbalanced():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 4), columns=list("abcd"))
    y = pd.Series([0] * 180 + [1] * 20)  # 10% minority
    return X, y


def test_class_distribution_counts():
    y = pd.Series([0, 0, 0, 1])
    dist = resampling.class_distribution(y)
    assert dist.loc[0, "count"] == 3
    assert dist.loc[1, "count"] == 1
    assert dist.loc[1, "pct"] == 25.0


def test_smote_balances_training_set():
    X, y = _imbalanced()
    Xr, yr = resampling.resample(X, y, method="smote")
    counts = pd.Series(yr).value_counts()
    # SMOTE oversamples the minority to match the majority.
    assert counts[0] == counts[1]
    assert len(Xr) == len(yr)


def test_undersample_reduces_majority():
    X, y = _imbalanced()
    Xr, yr = resampling.resample(X, y, method="undersample")
    counts = pd.Series(yr).value_counts()
    assert counts[0] == counts[1] == 20


def test_resample_report_shape():
    X, y = _imbalanced()
    _, yr = resampling.resample(X, y, method="smote")
    report = resampling.resample_report(y, yr)
    assert "count_before" in report.columns
    assert "count_after" in report.columns


def test_invalid_method_raises():
    X, y = _imbalanced()
    try:
        resampling.resample(X, y, method="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
