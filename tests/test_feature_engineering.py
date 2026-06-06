"""Tests for feature engineering."""
import pandas as pd

from src import feature_engineering as fe


def _sample():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 2],
            "device_id": ["D1", "D1", "D2"],
            "signup_time": pd.to_datetime(
                ["2015-01-01 00:00:00", "2015-01-01 00:00:00", "2015-01-01 00:00:00"]
            ),
            "purchase_time": pd.to_datetime(
                ["2015-01-01 05:00:00", "2015-01-01 06:00:00", "2015-01-03 00:00:00"]
            ),
        }
    )


def test_time_features():
    out = fe.add_time_features(_sample())
    assert out["hour_of_day"].tolist() == [5, 6, 0]
    assert out["day_of_week"].iloc[0] == 3  # 2015-01-01 was a Thursday
    assert out["time_since_signup"].iloc[0] == 5.0  # 5 hours


def test_frequency_features():
    out = fe.add_frequency_features(_sample())
    # Device D1 used by users {1, 2} -> 2 distinct users.
    d1 = out[out["device_id"] == "D1"]
    assert d1["device_user_count"].iloc[0] == 2
    assert d1["device_transaction_count"].iloc[0] == 2


def test_velocity_window():
    out = fe.add_velocity_features(_sample(), window_hours=24.0)
    # D1's two purchases are 1h apart -> the second has velocity 2.
    d1 = out[out["device_id"] == "D1"].sort_values("purchase_time")
    assert d1["device_velocity_24h"].tolist() == [1, 2]
    # D2 has a single purchase -> velocity 1.
    d2 = out[out["device_id"] == "D2"]
    assert d2["device_velocity_24h"].iloc[0] == 1
