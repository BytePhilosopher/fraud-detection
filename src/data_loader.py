"""Thin loaders for the raw datasets.

Each loader parses dates / dtypes at read time so downstream code receives a
correctly typed frame. Loaders raise a clear error if the file is absent, since
the IP-country and credit-card files must be supplied by the user.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset at {path}. Place the raw file in data/raw/ "
            "(see README) before running."
        )
    return path


def load_fraud_data(path: Path | None = None) -> pd.DataFrame:
    """Load Fraud_Data.csv with timestamps parsed to datetime."""
    path = _require(path or config.FRAUD_DATA_RAW)
    return pd.read_csv(path, parse_dates=config.FRAUD_DATETIME_COLS)


def load_ip_country(path: Path | None = None) -> pd.DataFrame:
    """Load the IP-range -> country lookup table."""
    path = _require(path or config.IP_COUNTRY_RAW)
    df = pd.read_csv(path)
    # Bounds occasionally ship as floats; integer bounds make range lookup exact.
    for col in ("lower_bound_ip_address", "upper_bound_ip_address"):
        df[col] = df[col].astype("int64")
    return df


def load_creditcard(path: Path | None = None) -> pd.DataFrame:
    """Load the credit-card transactions dataset."""
    path = _require(path or config.CREDITCARD_RAW)
    return pd.read_csv(path)
