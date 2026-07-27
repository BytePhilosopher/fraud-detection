"""Model construction for Task 2 — baseline and ensemble classifiers.

Design rules that the whole task hangs on
-----------------------------------------
1. **Split first, fit everything else inside.** The stratified train/test split
   happens on the *raw* feature frame. Scaling, one-hot encoding and SMOTE all
   live inside an ``imblearn`` ``Pipeline`` so they are re-fitted from scratch on
   every training fold. Nothing derived from the test set (or from a validation
   fold) can leak into a fitted transformer.
2. **SMOTE is a pipeline step, not a preprocessing step.** ``imblearn``'s
   pipeline calls samplers during ``fit`` only — never during ``transform`` /
   ``predict``. That is the mechanism that keeps the *evaluated* data at its real
   world class balance while the *trained* model still sees a balanced problem.
3. **Tuning optimises AUC-PR.** With 0.17%–9% positives, accuracy and even
   ROC-AUC are dominated by the majority class; average precision is the metric
   that actually tracks fraud-catching ability.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .config import RANDOM_STATE
from .transform import build_preprocessor

# --------------------------------------------------------------------------- #
# Feature schemas
# --------------------------------------------------------------------------- #
# Fraud_Data: engineered numerics from src.feature_engineering + the raw
# categoricals + the geolocated `country`. Country is high-cardinality (182
# levels), so the preprocessor groups rare levels rather than one-hot encoding
# a long tail of near-empty columns (see FRAUD_MIN_FREQUENCY).
FRAUD_NUMERIC_COLS = [
    "purchase_value",
    "age",
    "hour_of_day",
    "day_of_week",
    "time_since_signup",
    "user_transaction_count",
    "device_transaction_count",
    "device_user_count",
    "device_velocity_24h",
]
FRAUD_CATEGORICAL_COLS = ["source", "browser", "sex", "country"]

# Levels seen in <1% of training rows are folded into a single "infrequent"
# bucket. Keeps the design matrix compact and stops one-hot columns that are
# almost entirely zero from adding variance to the linear baseline.
FRAUD_MIN_FREQUENCY = 0.01


def creditcard_feature_cols(df: pd.DataFrame, target: str = "Class") -> list[str]:
    """All columns except the target — the credit-card set is already numeric PCA."""
    return [c for c in df.columns if c != target]


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_features_target(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target column."""
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in frame")
    cols = feature_cols if feature_cols is not None else [
        c for c in df.columns if c != target
    ]
    return df[cols].copy(), df[target].copy()


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Stratified train/test split — preserves the fraud rate in both halves.

    With 473 frauds in 283k credit-card rows, an unstratified split can easily
    hand one side a materially different positive rate; stratifying makes the
    test-set metrics comparable to production prevalence.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def build_logistic_regression(**kwargs: Any) -> LogisticRegression:
    """Interpretable linear baseline.

    ``lbfgs`` + L2 handles the wide one-hot matrix fine; ``max_iter`` is raised
    because the SMOTE-expanded training set converges slowly at the default 100.
    """
    params: dict[str, Any] = dict(
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    params.update(kwargs)
    return LogisticRegression(**params)


def build_xgboost(scale_pos_weight: float | None = None, **kwargs: Any):
    """Gradient-boosted trees (XGBoost) — the ensemble challenger.

    ``scale_pos_weight`` is the class-weighting alternative to SMOTE: it
    reweights the gradient contribution of positives instead of synthesising new
    rows. Pass it (and build the pipeline with ``sampler=None``) to compare the
    two imbalance strategies on equal footing.
    """
    from xgboost import XGBClassifier  # imported lazily: optional dependency

    params: dict[str, Any] = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    params.update(kwargs)
    return XGBClassifier(**params)


def positive_class_weight(y) -> float:
    """n_negative / n_positive — the standard ``scale_pos_weight`` value."""
    s = pd.Series(y)
    pos = int((s == 1).sum())
    neg = int((s == 0).sum())
    if pos == 0:
        raise ValueError("no positive-class rows: cannot compute a weight")
    return neg / pos


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def build_pipeline(
    estimator,
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    sampler: str | None = "smote",
    scaler: str = "standard",
    min_frequency: float | None = None,
) -> ImbPipeline:
    """Assemble preprocess -> (resample) -> estimate as one fittable object.

    Parameters
    ----------
    sampler : ``"smote"`` inserts SMOTE between preprocessing and the estimator;
        ``None`` skips resampling (use with ``class_weight`` /
        ``scale_pos_weight`` instead).

    Because the whole thing is a single estimator, ``cross_validate`` and
    ``GridSearchCV`` re-fit the scaler, the encoder *and* SMOTE on each inner
    training fold — the only leakage-free way to cross-validate resampled data.
    """
    pre = build_preprocessor(
        numeric_cols,
        categorical_cols or [],
        scaler=scaler,
        min_frequency=min_frequency,
    )
    steps: list[tuple[str, Any]] = [("preprocess", pre)]
    if sampler is not None:
        if sampler.lower() != "smote":
            raise ValueError("sampler must be 'smote' or None")
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


def fraud_pipeline(estimator, sampler: str | None = "smote") -> ImbPipeline:
    """Pipeline pre-wired with the Fraud_Data feature schema."""
    return build_pipeline(
        estimator,
        FRAUD_NUMERIC_COLS,
        FRAUD_CATEGORICAL_COLS,
        sampler=sampler,
        min_frequency=FRAUD_MIN_FREQUENCY,
    )


def creditcard_pipeline(
    estimator, numeric_cols: list[str], sampler: str | None = "smote"
) -> ImbPipeline:
    """Pipeline for the all-numeric credit-card matrix (no categoricals)."""
    return build_pipeline(estimator, numeric_cols, [], sampler=sampler)


# --------------------------------------------------------------------------- #
# Hyperparameter tuning
# --------------------------------------------------------------------------- #
#: Deliberately small grid. Each candidate is refit on every CV fold *including*
#: SMOTE, so a 100-point search would cost hours for a fraction of a point of
#: AUC-PR. n_estimators / max_depth are the two knobs that move the metric most.
XGB_PARAM_GRID = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [3, 6],
    "clf__learning_rate": [0.05, 0.2],
}

LOGREG_PARAM_GRID = {"clf__C": [0.1, 1.0, 10.0]}


def tune(
    pipeline: ImbPipeline,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 3,
    scoring: str = "average_precision",
    verbose: int = 0,
) -> GridSearchCV:
    """Grid-search ``pipeline`` on the training set, scoring by AUC-PR.

    ``cv=3`` (not 5) for the search: the search is only choosing between
    candidates, and the winner is re-validated properly with 5-fold CV
    afterwards. Returns the fitted search object (``best_estimator_`` is refit
    on the full training set).
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        cv=splitter,
        n_jobs=1,  # the estimators already parallelise internally
        refit=True,
        verbose=verbose,
    )
    search.fit(X, y)
    return search


def search_results_frame(search: GridSearchCV) -> pd.DataFrame:
    """Tidy ranked view of a completed grid search."""
    cols = [c for c in search.cv_results_ if c.startswith("param_")]
    out = pd.DataFrame(search.cv_results_)[
        cols + ["mean_test_score", "std_test_score", "rank_test_score"]
    ]
    return out.sort_values("rank_test_score").reset_index(drop=True)
