# Fraud Detection

Machine-learning pipeline for detecting fraud in **e-commerce transactions**
(`Fraud_Data.csv`) and **bank credit-card transactions** (`creditcard.csv`),
with IP-based geolocation enrichment. Both targets are highly imbalanced, which
drives the resampling strategy and evaluation metrics throughout.

## Project status

**Task 1 — Data Analysis & Preprocessing: complete.** Cleaned datasets,
EDA, geolocation integration, feature engineering, scaling/encoding, and
SMOTE-based imbalance handling are implemented, tested, and executed.
See [`reports/task1-report.md`](reports/task1-report.md).

**Task 2 — Model Building & Training: complete.** Stratified splits, a Logistic
Regression baseline, a tuned XGBoost ensemble, 5-fold stratified CV, threshold
selection and model selection. See
[`reports/task2-report.md`](reports/task2-report.md).

**Task 3 — SHAP explainability:** scaffolded.

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
│   ├── resampling.py      # SMOTE / undersampling (train only)
│   ├── modeling.py        # stratified split, estimators, leakage-safe pipelines, tuning
│   └── evaluation.py      # AUC-PR / F1 / confusion matrix, CV, threshold selection
├── tests/                 # pytest suite for src/ (55 tests)
├── scripts/               # build_notebooks.py, train_models.py
├── models/                # saved pipelines + thresholds (gitignored)
├── reports/               # task reports, metric tables
│   └── figures/           # generated EDA & evaluation figures
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
python scripts/train_models.py             # Task 2: train, tune & compare (~26 min)
python scripts/train_models.py --quick     # 10% sample smoke test (~2 min)
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

## Task 2 highlights

### Protocol (identical for both datasets)
Stratified 80/20 split → grid search on the training half (3-fold, scored by
AUC-PR) → 5-fold stratified CV of the tuned pipeline → decision threshold chosen
from out-of-fold *training* probabilities → a single evaluation on the untouched
test set.

Scaling, encoding **and** SMOTE are steps inside an `imblearn` pipeline, so every
fold refits its own transformers and sampler. `imblearn` runs samplers on `fit`
only, so the test set is always scored at its real 9.36% / 0.167% fraud rate —
resampling before cross-validating is the standard way this task gets silently
corrupted.

### Models compared
| Key | Model | Imbalance strategy |
|---|---|---|
| `logreg_smote` | Logistic Regression (L2) | SMOTE |
| `xgb_smote` | XGBoost | SMOTE |
| `xgb_weighted` | XGBoost | `scale_pos_weight = n_neg/n_pos` |

### Results — 5-fold CV AUC-PR (training half)
| Model | Fraud_Data | creditcard |
|---|---|---|
| Logistic Regression + SMOTE | 0.6823 ± 0.0073 | 0.7529 ± 0.0262 |
| XGBoost + SMOTE | 0.7212 ± 0.0061 | 0.8486 ± 0.0314 |
| **XGBoost + `scale_pos_weight`** | **0.7226 ± 0.0059** | **0.8531 ± 0.0271** |

### Selected model
**XGBoost + `scale_pos_weight`** on both datasets — highest AUC-PR in CV *and* on
the test set (0.714 / 0.822), and it dominates the baseline across the whole
precision-recall curve. Choosing AUC-PR as the primary metric is what made the
difference visible: on the credit-card data the logistic baseline reports
ROC-AUC **0.963** while flagging **1,481 legitimate transactions to catch 83
frauds** at the default threshold (F1 0.10).

Interpretability was the real trade-off, and it was conceded deliberately: the
credit-card features are anonymised PCA components, so a linear model's
coefficients carry no business meaning anyway, and SHAP (Task 3) recovers both
global importance and per-transaction explanations for the ensemble. The
baseline stays in the repo as the bar any future model must clear.

Full writeup, error profiles and deployment notes:
[`reports/task2-report.md`](reports/task2-report.md).

## Testing & CI

`pytest` covers cleaning, geolocation (incl. range edge cases), feature
engineering, transformation, resampling, model construction and evaluation — 55
tests. One of them, `test_smote_does_not_change_prediction_row_count`, is a
leakage guard: it fails if resampling ever escapes the `fit` path into
`predict`. GitHub Actions runs the suite on every push/PR
(`.github/workflows/unittests.yml`).
