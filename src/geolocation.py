"""IP-address geolocation: convert IPs to integers and map to country.

The IP-country table stores *ranges* (lower/upper bound). A naive join is
O(n*m); instead we sort the ranges once and use a binary search
(``np.searchsorted``) to locate the candidate range for each transaction in
O(n log m), then validate the upper bound.
"""
from __future__ import annotations

import ipaddress

import numpy as np
import pandas as pd

UNKNOWN_COUNTRY = "Unknown"


def ip_to_int(ip: float | int | str) -> int | float:
    """Convert an IP address to its integer representation.

    Accepts the float form found in Fraud_Data (e.g. 732758368.79972),
    dotted-quad strings ('1.2.3.4'), and plain integers. Returns ``np.nan`` for
    unparseable values so callers can decide how to handle them.
    """
    if pd.isna(ip):
        return np.nan
    # Dotted-quad string (e.g. '1.2.3.4') -> integer
    if isinstance(ip, str) and ip.strip().count(".") == 3:
        try:
            return int(ipaddress.IPv4Address(ip.strip()))
        except (ipaddress.AddressValueError, ValueError):
            return np.nan
    # Numeric (float/int) or numeric-string -> truncate fractional part
    try:
        return int(float(ip))
    except (TypeError, ValueError):
        return np.nan


def add_ip_integer(df: pd.DataFrame, ip_col: str = "ip_address",
                   out_col: str = "ip_int") -> pd.DataFrame:
    """Add an integer IP column derived from ``ip_col``."""
    df = df.copy()
    df[out_col] = df[ip_col].apply(ip_to_int).astype("Int64")
    return df


def merge_country(
    fraud_df: pd.DataFrame,
    ip_country_df: pd.DataFrame,
    ip_int_col: str = "ip_int",
) -> pd.DataFrame:
    """Attach a ``country`` column via range-based lookup.

    Parameters
    ----------
    fraud_df : frame containing an integer IP column (``ip_int_col``).
    ip_country_df : the lookup table with lower/upper bound + country.
    """
    df = fraud_df.copy()

    lookup = ip_country_df.sort_values("lower_bound_ip_address").reset_index(drop=True)
    lowers = lookup["lower_bound_ip_address"].to_numpy()
    uppers = lookup["upper_bound_ip_address"].to_numpy()
    countries = lookup["country"].to_numpy()

    ips = df[ip_int_col].to_numpy()

    # Candidate range index = last lower-bound <= ip.
    idx = np.searchsorted(lowers, ips, side="right") - 1

    result = np.full(len(df), UNKNOWN_COUNTRY, dtype=object)
    valid = idx >= 0
    # Among candidates, keep only those whose ip also falls under upper bound
    # and whose ip itself is not missing.
    cand = idx.copy()
    cand[~valid] = 0  # placeholder; masked out below
    within_upper = ips <= uppers[cand]
    not_na = ~pd.isna(ips)
    matched = valid & within_upper & not_na
    result[matched] = countries[cand[matched]]

    df["country"] = result
    return df


def add_geolocation(
    fraud_df: pd.DataFrame,
    ip_country_df: pd.DataFrame,
    ip_col: str = "ip_address",
) -> pd.DataFrame:
    """Convenience wrapper: IP -> int -> country in one call."""
    df = add_ip_integer(fraud_df, ip_col=ip_col)
    df = merge_country(df, ip_country_df)
    return df


def fraud_rate_by_country(df: pd.DataFrame, target: str = "class",
                          min_count: int = 50) -> pd.DataFrame:
    """Fraud rate and volume per country (countries with >= ``min_count`` rows)."""
    grouped = (
        df.groupby("country")[target]
        .agg(transactions="count", frauds="sum")
        .assign(fraud_rate=lambda x: (x["frauds"] / x["transactions"]).round(4))
    )
    return grouped[grouped["transactions"] >= min_count].sort_values(
        "fraud_rate", ascending=False
    )
