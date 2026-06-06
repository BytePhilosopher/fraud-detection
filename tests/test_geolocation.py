"""Tests for IP conversion and range-based country lookup."""
import ipaddress

import numpy as np
import pandas as pd

from src import geolocation as geo


def test_ip_to_int_float():
    # Fraud_Data ships IPs as floats; fractional part is truncated.
    assert geo.ip_to_int(732758368.79972) == 732758368


def test_ip_to_int_dotted_quad():
    assert geo.ip_to_int("1.2.3.4") == int(ipaddress.IPv4Address("1.2.3.4"))


def test_ip_to_int_handles_nan():
    assert np.isnan(geo.ip_to_int(np.nan))
    assert np.isnan(geo.ip_to_int("not-an-ip"))


def test_merge_country_range_lookup():
    fraud = pd.DataFrame({"ip_int": [50, 150, 250, 9999]})
    lookup = pd.DataFrame(
        {
            "lower_bound_ip_address": [0, 100, 200],
            "upper_bound_ip_address": [99, 199, 299],
            "country": ["Alpha", "Beta", "Gamma"],
        }
    )
    out = geo.merge_country(fraud, lookup)
    assert out["country"].tolist() == ["Alpha", "Beta", "Gamma", geo.UNKNOWN_COUNTRY]


def test_merge_country_gap_between_ranges():
    # IP that falls in a gap (no range covers it) -> Unknown.
    fraud = pd.DataFrame({"ip_int": [150]})
    lookup = pd.DataFrame(
        {
            "lower_bound_ip_address": [0, 200],
            "upper_bound_ip_address": [99, 299],
            "country": ["Alpha", "Gamma"],
        }
    )
    out = geo.merge_country(fraud, lookup)
    assert out["country"].iloc[0] == geo.UNKNOWN_COUNTRY


def test_fraud_rate_by_country():
    df = pd.DataFrame(
        {
            "country": ["A"] * 60 + ["B"] * 60,
            "class": [1] * 30 + [0] * 30 + [0] * 60,
        }
    )
    rates = geo.fraud_rate_by_country(df, min_count=50)
    assert rates.loc["A", "fraud_rate"] == 0.5
    assert rates.loc["B", "fraud_rate"] == 0.0
