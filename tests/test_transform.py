"""Tests for scaling + encoding preprocessor."""
import numpy as np
import pandas as pd

from src import transform


def _frame():
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0],
            "num2": [10.0, 20.0, 30.0, 40.0],
            "cat": ["a", "b", "a", "b"],
        }
    )


def test_standard_scaler_zero_mean():
    pre = transform.build_preprocessor(["num1", "num2"], ["cat"], scaler="standard")
    out = transform.fit_transform_frame(pre, _frame())
    # Scaled numeric columns should have ~zero mean.
    assert np.isclose(out["num__num1"].mean(), 0.0, atol=1e-9)


def test_one_hot_creates_category_columns():
    pre = transform.build_preprocessor(["num1"], ["cat"], scaler="minmax")
    out = transform.fit_transform_frame(pre, _frame())
    cat_cols = [c for c in out.columns if c.startswith("cat__")]
    assert len(cat_cols) == 2  # 'a' and 'b'


def test_transform_uses_fitted_state():
    pre = transform.build_preprocessor(["num1"], ["cat"], scaler="standard")
    train = _frame()
    transform.fit_transform_frame(pre, train)
    # Unseen category must not error (handle_unknown='ignore').
    test = pd.DataFrame({"num1": [2.5], "cat": ["z"]})
    out = transform.transform_frame(pre, test)
    assert out.shape[0] == 1


def test_invalid_scaler_raises():
    try:
        transform.build_preprocessor(["num1"], ["cat"], scaler="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
