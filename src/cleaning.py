"""Data-cleaning utilities: dtypes, duplicates, missing values.

The functions are intentionally small and composable so notebooks can show the
effect of each step, and so they can be unit-tested in isolation.
"""
from __future__ import annotations

import pandas as pd

from . import config


def correct_fraud_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce Fraud_Data columns to their semantically correct dtypes."""
    df = df.copy()
    for col in config.FRAUD_DATETIME_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in config.FRAUD_CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    # ip_address arrives as float; keep the integer part for range lookup.
    if "ip_address" in df.columns:
        df["ip_address"] = df["ip_address"].astype("float64")
    return df


def drop_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Remove fully duplicated rows (or duplicates on a key subset)."""
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column null counts and percentages, sorted desc."""
    counts = df.isnull().sum()
    pct = (counts / len(df) * 100).round(3)
    report = pd.DataFrame({"missing": counts, "missing_pct": pct})
    return report[report["missing"] > 0].sort_values("missing", ascending=False)


def handle_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Conservative missing-value policy.

    - Columns with > ``threshold`` fraction missing are dropped (too sparse to
      impute reliably).
    - Remaining numeric NaNs -> median (robust to skew/outliers).
    - Remaining categorical/object NaNs -> mode.

    For the supplied datasets there are no missing values, so this acts as a
    documented safeguard against future / re-pulled data.
    """
    df = df.copy()
    n = len(df)
    too_sparse = [c for c in df.columns if df[c].isnull().sum() / n > threshold]
    df = df.drop(columns=too_sparse)

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end cleaning pipeline for Fraud_Data."""
    df = correct_fraud_dtypes(df)
    df = drop_duplicates(df)
    df = handle_missing_values(df)
    return df


def clean_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning pipeline for the credit-card dataset.

    The PCA features are already numeric; the only realistic data-quality issue
    is exact duplicate transactions, which are dropped.
    """
    df = drop_duplicates(df)
    df = handle_missing_values(df)
    return df
