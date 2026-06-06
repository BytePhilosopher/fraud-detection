"""Feature engineering for Fraud_Data.

Two families of features:
  * Time-based   : hour_of_day, day_of_week, time_since_signup.
  * Frequency /  : per-user and per-device transaction counts and a short-window
    velocity        velocity flag that captures rapid repeat activity often seen
                    in card-testing / account-takeover fraud.
"""
from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal features from signup / purchase timestamps."""
    df = df.copy()
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek  # 0 = Monday
    # Duration between account creation and first purchase, in hours.
    delta = df["purchase_time"] - df["signup_time"]
    df["time_since_signup"] = delta.dt.total_seconds() / 3600.0
    # A non-positive gap indicates the purchase landed at/after signup oddly fast;
    # near-zero gaps are a strong automated-fraud signal, so keep the raw value.
    return df


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Count transactions sharing a user or a device.

    A device used by many distinct accounts, or repeated purchases from one
    user, are classic fraud-ring signals.
    """
    df = df.copy()
    df["user_transaction_count"] = df.groupby("user_id")["user_id"].transform("count")
    df["device_transaction_count"] = df.groupby("device_id")["device_id"].transform(
        "count"
    )
    # Distinct users seen on each device (device sharing across accounts).
    df["device_user_count"] = df.groupby("device_id")["user_id"].transform("nunique")
    return df


def add_velocity_features(df: pd.DataFrame, window_hours: float = 24.0) -> pd.DataFrame:
    """Per-device purchase velocity within a rolling time window.

    For each device we count how many purchases from the same device occurred in
    the preceding ``window_hours`` (inclusive of the current transaction). High
    velocity is indicative of automated abuse. Implemented with pandas'
    time-based ``groupby.rolling`` for vectorized speed.
    """
    df = df.copy()
    col = f"device_velocity_{int(window_hours)}h"
    window = pd.Timedelta(hours=window_hours)

    # Sort by (device, time); groupby.rolling preserves this order, so the
    # resulting counts align positionally with df_sorted.
    df_sorted = df.sort_values(["device_id", "purchase_time"])
    counts = (
        df_sorted.set_index("purchase_time")
        .groupby("device_id")["user_id"]
        .rolling(window)
        .count()
        .to_numpy()
    )
    df_sorted[col] = counts.astype("int64")
    # Restore original row order.
    df[col] = df_sorted[col].reindex(df.index)
    return df


def engineer_fraud_features(
    df: pd.DataFrame, velocity_window_hours: float = 24.0
) -> pd.DataFrame:
    """Run the full Fraud_Data feature-engineering pipeline."""
    df = add_time_features(df)
    df = add_frequency_features(df)
    df = add_velocity_features(df, window_hours=velocity_window_hours)
    return df
