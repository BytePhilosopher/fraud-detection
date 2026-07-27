"""Tests for model construction, splitting and tuning (Task 2).

The properties worth locking down are the ones that silently corrupt results if
they break: stratification, and SMOTE being confined to ``fit``.
"""
import numpy as np
import pandas as pd
import pytest

from src import modeling


def _imbalanced(n=400, minority=40):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {
            "num1": rng.randn(n),
            "num2": rng.randn(n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    y = pd.Series([0] * (n - minority) + [1] * minority, name="class")
    # Give the minority a real signal so metrics are not pure noise.
    X.loc[y == 1, "num1"] += 3.0
    return X, y


NUM = ["num1", "num2"]
CAT = ["cat"]


def test_split_features_target_separates_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "class": [0, 1]})
    X, y = modeling.split_features_target(df, "class")
    assert list(X.columns) == ["a", "b"]
    assert y.tolist() == [0, 1]


def test_split_features_target_honours_explicit_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "class": [0, 1]})
    X, _ = modeling.split_features_target(df, "class", ["b"])
    assert list(X.columns) == ["b"]


def test_split_features_target_missing_target_raises():
    with pytest.raises(KeyError):
        modeling.split_features_target(pd.DataFrame({"a": [1]}), "class")


def test_stratified_split_preserves_class_ratio():
    X, y = _imbalanced()
    X_train, X_test, y_train, y_test = modeling.stratified_split(X, y, test_size=0.25)
    assert len(X_train) == 300 and len(X_test) == 100
    # 10% minority must survive in both halves.
    assert y_train.mean() == pytest.approx(y.mean(), abs=0.01)
    assert y_test.mean() == pytest.approx(y.mean(), abs=0.01)


def test_positive_class_weight_is_neg_over_pos():
    y = pd.Series([0] * 90 + [1] * 10)
    assert modeling.positive_class_weight(y) == pytest.approx(9.0)


def test_positive_class_weight_no_positives_raises():
    with pytest.raises(ValueError):
        modeling.positive_class_weight(pd.Series([0, 0, 0]))


def test_pipeline_has_smote_step_by_default():
    pipe = modeling.build_pipeline(
        modeling.build_logistic_regression(), NUM, CAT, sampler="smote"
    )
    assert list(dict(pipe.steps)) == ["preprocess", "smote", "clf"]


def test_pipeline_without_sampler_skips_smote():
    pipe = modeling.build_pipeline(
        modeling.build_logistic_regression(), NUM, CAT, sampler=None
    )
    assert "smote" not in dict(pipe.steps)


def test_invalid_sampler_raises():
    with pytest.raises(ValueError):
        modeling.build_pipeline(
            modeling.build_logistic_regression(), NUM, CAT, sampler="bogus"
        )


def test_smote_does_not_change_prediction_row_count():
    """The leakage guard: SMOTE runs on fit, never on predict.

    If resampling leaked into the transform path, `predict` would return more
    rows than it was given and every test-set metric would be meaningless.
    """
    X, y = _imbalanced()
    pipe = modeling.build_pipeline(modeling.build_logistic_regression(), NUM, CAT)
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(X)
    assert pipe.predict_proba(X).shape == (len(X), 2)


def test_pipeline_fits_and_scores_better_than_chance():
    X, y = _imbalanced()
    pipe = modeling.build_pipeline(modeling.build_logistic_regression(), NUM, CAT)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    # Minority rows carry a +3 sigma shift; mean score must separate the classes.
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_all_numeric_pipeline_needs_no_categoricals():
    X, y = _imbalanced()
    pipe = modeling.build_pipeline(
        modeling.build_logistic_regression(), NUM, [], sampler="smote"
    )
    pipe.fit(X[NUM], y)
    assert len(pipe.predict(X[NUM])) == len(X)


def test_min_frequency_collapses_rare_levels():
    X, y = _imbalanced()
    X = X.copy()
    X.loc[X.index[:3], "cat"] = "rare"  # 3/400 = 0.75% < 5%
    pipe = modeling.build_pipeline(
        modeling.build_logistic_regression(), NUM, CAT, min_frequency=0.05
    )
    pipe.fit(X, y)
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    assert not any(n.endswith("rare") for n in names)


def test_fraud_pipeline_uses_fraud_schema():
    pipe = modeling.fraud_pipeline(modeling.build_logistic_regression())
    cat_cols = pipe.named_steps["preprocess"].transformers[1][2]
    assert "country" in cat_cols
    assert "time_since_signup" in pipe.named_steps["preprocess"].transformers[0][2]


def test_creditcard_feature_cols_drops_target():
    df = pd.DataFrame({"Time": [1], "V1": [0.5], "Class": [0]})
    assert modeling.creditcard_feature_cols(df, "Class") == ["Time", "V1"]


def test_tune_returns_fitted_search_with_best_estimator():
    X, y = _imbalanced()
    pipe = modeling.build_pipeline(modeling.build_logistic_regression(), NUM, CAT)
    search = modeling.tune(pipe, {"clf__C": [0.1, 1.0]}, X, y, cv=2)
    assert search.best_params_["clf__C"] in (0.1, 1.0)
    # refit=True: the winner is usable immediately.
    assert len(search.best_estimator_.predict(X)) == len(X)


def test_search_results_frame_is_ranked():
    X, y = _imbalanced()
    pipe = modeling.build_pipeline(modeling.build_logistic_regression(), NUM, CAT)
    search = modeling.tune(pipe, {"clf__C": [0.1, 1.0]}, X, y, cv=2)
    frame = modeling.search_results_frame(search)
    assert frame.loc[0, "rank_test_score"] == 1
    assert "mean_test_score" in frame.columns


def test_build_xgboost_accepts_scale_pos_weight():
    pytest.importorskip("xgboost")
    clf = modeling.build_xgboost(scale_pos_weight=9.0, n_estimators=10)
    assert clf.get_params()["scale_pos_weight"] == 9.0
    assert clf.get_params()["n_estimators"] == 10
