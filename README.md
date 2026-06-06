# Fraud Detection

Machine-learning pipeline for detecting fraud in **e-commerce transactions**
(`Fraud_Data.csv`) and **bank credit-card transactions** (`creditcard.csv`),
with IP-based geolocation enrichment. Both targets are highly imbalanced, which
drives the resampling strategy and evaluation metrics throughout.

## Project status

**Task 1 — Data Analysis & Preprocessing: complete.** Cleaned datasets,
EDA, geolocation integration, feature engineering, scaling/encoding, and
SMOTE-based imbalance handling are implemented, tested, and executed.
Modeling (Task 2) and SHAP explainability (Task 3) are scaffolded.

## Repository layout

```
fraud-detection/
├── data/                  # gitignored
│   ├── raw/               # Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv
│   └── processed/         # fraud_features.csv (generated)
├── notebooks/             # narrative analysis (see notebooks/README.md)
├── src/                   # reusable, unit-tested modules
│   ├── config.py          # paths & constants
│   ├── data_loader.py     # typed loaders
│   ├── cleaning.py        # dtypes, duplicates, missing values
│   ├── geolocation.py     # IP→int, range-based country merge
│   ├── feature_engineering.py  # time, frequency, velocity features
│   ├── transform.py       # scaling + one-hot encoding (ColumnTransformer)
│   └── resampling.py      # SMOTE / undersampling (train only)
├── tests/                 # pytest suite for src/
├── scripts/               # build_notebooks.py
├── models/                # saved artifacts (gitignored)
├── reports/figures/       # generated EDA figures
└── .github/workflows/     # CI: run tests on push/PR
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the three raw CSVs in `data/raw/`. Then:

```bash
pytest                                     # run the test suite
python scripts/build_notebooks.py          # generate notebooks
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## Task 1 highlights

### Data cleaning
- **Fraud_Data:** timestamps parsed to `datetime`, `source/browser/sex` cast to
  `category`, `ip_address` kept numeric for lookup. No nulls; no exact duplicate
  rows. Shared `device_id`s are *retained* — device sharing is a fraud signal.
- **creditcard:** 1,081 exact duplicate transactions dropped; no missing values.
- Documented missing-value policy (`handle_missing_values`): drop columns >50%
  null, median-impute numeric, mode-impute categorical — a safeguard for future
  data pulls.

### Class imbalance
| Dataset | Fraud rate |
|---|---|
| Fraud_Data | ~9.4% |
| creditcard | ~0.17% |

Accuracy is useless here (a trivial classifier scores 90.6% / 99.83%). Primary
metric is **AUC-PR**, supported by recall, precision and F1.

### Geolocation integration
IP floats → integers, then mapped to country via **range-based lookup**
(`np.searchsorted`, O(n log m)) against 138k IP ranges. ~85% match rate. Fraud
rate varies markedly by country (e.g. Ecuador/Tunisia/Peru ≈ 26% vs ~9%
baseline).

### Feature engineering (Fraud_Data)
- **Time:** `hour_of_day`, `day_of_week`, `time_since_signup` (hours; near-zero
  gaps flag automated fraud).
- **Frequency/velocity:** per-user & per-device transaction counts,
  `device_user_count` (distinct accounts per device), and `device_velocity_24h`
  (rolling 24h purchase count per device).

### Transformation & imbalance handling
- `StandardScaler` (numerics) + `OneHotEncoder` (categoricals) in a single
  `ColumnTransformer`, **fit on the training fold only**.
- **SMOTE** chosen over under/over-sampling: undersampling would discard most
  legitimate data; naive oversampling overfits via duplication; SMOTE
  interpolates synthetic minority samples. Applied to the **training set only** —
  the test set keeps its real-world distribution. Class distribution before/after
  is documented in `feature-engineering.ipynb`.

## Testing & CI

`pytest` covers cleaning, geolocation (incl. range edge cases), feature
engineering, transformation, and resampling. GitHub Actions runs the suite on
every push/PR (`.github/workflows/unittests.yml`).
