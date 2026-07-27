"""Feature transformation: scaling numeric features and encoding categoricals.

We expose a scikit-learn ``ColumnTransformer`` so the exact same fitted
transformation can be reused at inference time, and so it slots directly into a
modeling ``Pipeline`` (fit on train only -> no leakage).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scaler: str = "standard",
    min_frequency: float | int | None = None,
) -> ColumnTransformer:
    """Create a ColumnTransformer that scales numerics and one-hot encodes cats.

    Parameters
    ----------
    scaler : "standard" (StandardScaler) or "minmax" (MinMaxScaler).
    min_frequency : if given, categorical levels rarer than this (a proportion
        when < 1, else a count) are collapsed into one "infrequent" column.
        Needed for high-cardinality fields such as ``country`` (182 levels),
        where a full one-hot expansion is mostly near-empty columns.
    """
    if scaler == "standard":
        num_scaler = StandardScaler()
    elif scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        num_scaler = MinMaxScaler()
    else:
        raise ValueError("scaler must be 'standard' or 'minmax'")

    transformers = [("num", num_scaler, numeric_cols)]
    # An empty column list would still emit a fitted-but-useless encoder, so
    # skip the step entirely for all-numeric data (e.g. the credit-card PCA set).
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist"
                    if min_frequency is not None
                    else "ignore",
                    sparse_output=False,
                    min_frequency=min_frequency,
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_transform_frame(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Fit the preprocessor on ``df`` and return a transformed DataFrame.

    Column names are recovered from the fitted transformer so the output stays
    interpretable (important for SHAP later).
    """
    arr = preprocessor.fit_transform(df)
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame(arr, columns=names, index=df.index)


def transform_frame(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply an already-fitted preprocessor (e.g. to the test set)."""
    arr = preprocessor.transform(df)
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame(arr, columns=names, index=df.index)
