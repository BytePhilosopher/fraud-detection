"""Tests for cleaning utilities."""
import numpy as np
import pandas as pd

from src import cleaning


def test_drop_duplicates_removes_exact_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out = cleaning.drop_duplicates(df)
    assert len(out) == 2


def test_missing_value_report_only_lists_columns_with_nulls():
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [1, 2, 3]})
    report = cleaning.missing_value_report(df)
    assert "a" in report.index
    assert "b" not in report.index
    assert report.loc["a", "missing"] == 1


def test_handle_missing_values_imputes_numeric_median():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    out = cleaning.handle_missing_values(df)
    assert out["a"].isnull().sum() == 0
    assert out["a"].iloc[1] == 2.0  # median of [1, 3]


def test_handle_missing_values_drops_too_sparse_columns():
    df = pd.DataFrame({"keep": [1, 2, 3, 4], "drop": [np.nan, np.nan, np.nan, 1]})
    out = cleaning.handle_missing_values(df, threshold=0.5)
    assert "drop" not in out.columns
    assert "keep" in out.columns


def test_correct_fraud_dtypes_parses_dates_and_categories():
    df = pd.DataFrame(
        {
            "signup_time": ["2015-01-01 00:00:00"],
            "purchase_time": ["2015-01-02 00:00:00"],
            "source": ["SEO"],
            "browser": ["Chrome"],
            "sex": ["M"],
            "ip_address": [123.4],
        }
    )
    out = cleaning.correct_fraud_dtypes(df)
    assert pd.api.types.is_datetime64_any_dtype(out["signup_time"])
    assert isinstance(out["source"].dtype, pd.CategoricalDtype)
