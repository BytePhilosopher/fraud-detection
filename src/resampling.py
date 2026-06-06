"""Class-imbalance handling.

Critical rule: resampling is applied to the TRAINING SET ONLY. The test set must
keep its real-world (imbalanced) distribution, otherwise evaluation metrics are
optimistically biased.

Choice of technique
-------------------
We default to **SMOTE** (Synthetic Minority Over-sampling Technique):
  * Random undersampling discards the majority class — with frauds at <10% of
    Fraud_Data and <0.2% of the credit-card data, undersampling would throw away
    the vast majority of legitimate examples and the signal they carry.
  * Random oversampling duplicates minority rows, which encourages overfitting to
    those exact points.
  * SMOTE interpolates new synthetic minority samples between near neighbours,
    enriching the minority region of feature space without literal duplication.

For the extremely imbalanced credit-card set, ``SMOTEENN`` (SMOTE + Edited
Nearest Neighbours cleaning) is also provided as it removes ambiguous synthetic
points near the decision boundary.
"""
from __future__ import annotations

import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from .config import RANDOM_STATE


def class_distribution(y) -> pd.DataFrame:
    """Return absolute counts and proportions for each class label."""
    s = pd.Series(y)
    counts = s.value_counts().sort_index()
    pct = (counts / len(s) * 100).round(3)
    return pd.DataFrame({"count": counts, "pct": pct})


def resample(X, y, method: str = "smote"):
    """Resample (X, y) using the chosen strategy.

    method : "smote" | "undersample" | "smoteenn"
    Returns (X_resampled, y_resampled).
    """
    method = method.lower()
    if method == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    elif method == "smoteenn":
        sampler = SMOTEENN(random_state=RANDOM_STATE)
    else:
        raise ValueError("method must be 'smote', 'undersample', or 'smoteenn'")
    return sampler.fit_resample(X, y)


def resample_report(y_before, y_after) -> pd.DataFrame:
    """Side-by-side class distribution before vs after resampling."""
    before = class_distribution(y_before).add_suffix("_before")
    after = class_distribution(y_after).add_suffix("_after")
    return before.join(after, how="outer").fillna(0)
