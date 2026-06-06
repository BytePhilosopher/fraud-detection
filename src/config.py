"""Central configuration: paths and shared constants.

Keeping all filesystem locations in one place means notebooks, scripts, and
tests resolve data the same way regardless of the directory they run from.
"""
from __future__ import annotations

from pathlib import Path

# Project root = parent of the `src/` package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw input files
FRAUD_DATA_RAW = RAW_DIR / "Fraud_Data.csv"
IP_COUNTRY_RAW = RAW_DIR / "IpAddress_to_Country.csv"
CREDITCARD_RAW = RAW_DIR / "creditcard.csv"

# Processed outputs
FRAUD_FEATURES = PROCESSED_DIR / "fraud_features.csv"
CREDITCARD_CLEAN = PROCESSED_DIR / "creditcard_clean.csv"

RANDOM_STATE = 42

# Schema-level metadata for Fraud_Data, reused across modules.
FRAUD_DATETIME_COLS = ["signup_time", "purchase_time"]
FRAUD_CATEGORICAL_COLS = ["source", "browser", "sex"]
FRAUD_TARGET = "class"
CREDITCARD_TARGET = "Class"


def ensure_dirs() -> None:
    """Create output directories if they do not yet exist."""
    for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
