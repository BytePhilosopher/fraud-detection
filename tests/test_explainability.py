"""Tests for built-in importance and SHAP explanation helpers (Task 3)."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from src import explainability as xai, modeling as md

NUM = ["num1", "num2"]
CAT = ["cat"]


def _data(n=300, minority=40):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {
            "num1": rng.randn(n),
            "num2": rng.randn(n) * 10 + 50,  # different scale, so scaling is visible
            "cat": rng.choice(["a", "b"], n),
        }
    )
    y = pd.Series([0] * (n - minority) + [1] * minority)
    X.loc[y == 1, "num1"] += 3.0  # the only real signal
    return X, y


@pytest.fixture(scope="module")
def fitted():
    pytest.importorskip("xgboost")
    X, y = _data()
    pipe = md.build_pipeline(
        md.build_xgboost(n_estimators=25, max_depth=3), NUM, CAT, sampler=None
    )
    pipe.fit(X, y)
    return pipe, X, y


@pytest.fixture(scope="module")
def explanation(fitted):
    pytest.importorskip("shap")
    pipe, X, _ = fitted
    return xai.shap_explanation(pipe, X)


# --------------------------------------------------------------------------- #
# Pipeline bridging
# --------------------------------------------------------------------------- #
def test_preprocessed_frame_has_model_space_columns(fitted):
    pipe, X, _ = fitted
    frame = xai.preprocessed_frame(pipe, X)
    assert len(frame) == len(X)
    assert frame.shape[1] == len(pipe.named_steps["preprocess"].get_feature_names_out())
    assert list(frame.index) == list(X.index)


def test_tidy_names_strips_transformer_prefixes():
    assert xai.tidy_names(["num__age", "cat__source_SEO", "plain"]) == [
        "age",
        "source_SEO",
        "plain",
    ]


def test_display_frame_restores_original_units(fitted):
    pipe, X, _ = fitted
    model_frame = xai.preprocessed_frame(pipe, X)
    display = xai.display_frame(pipe, X, model_frame)
    # num2 is centred near 50 in the raw data but ~0 after standardisation.
    scaled_col = [c for c in model_frame.columns if c.endswith("num2")][0]
    assert abs(model_frame[scaled_col].mean()) < 0.5
    assert display[scaled_col].mean() == pytest.approx(X["num2"].mean())


def test_display_frame_leaves_one_hot_columns_alone(fitted):
    pipe, X, _ = fitted
    model_frame = xai.preprocessed_frame(pipe, X)
    display = xai.display_frame(pipe, X, model_frame)
    onehot = [c for c in model_frame.columns if "cat__" in c]
    assert onehot, "expected one-hot columns"
    for c in onehot:
        assert set(np.unique(display[c])) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# Built-in importance
# --------------------------------------------------------------------------- #
def test_builtin_importance_is_sorted_and_normalised(fitted):
    pipe, _, _ = fitted
    imp = xai.builtin_importance(pipe)
    assert imp.is_monotonic_decreasing
    assert imp.sum() == pytest.approx(1.0, abs=1e-6)
    assert "num1" in imp.index  # names are tidied


def test_builtin_importance_ranks_the_signal_feature_first(fitted):
    pipe, _, _ = fitted
    # num1 carries the only class signal, so it must dominate gain.
    assert xai.builtin_importance(pipe).index[0] == "num1"


def test_builtin_importance_weight_type_also_works(fitted):
    pipe, _, _ = fitted
    weight = xai.builtin_importance(pipe, importance_type="weight")
    assert weight.sum() == pytest.approx(1.0, abs=1e-6)
    assert len(weight) == len(xai.builtin_importance(pipe))


def test_builtin_importance_rejects_models_without_importances():
    pipe = md.build_pipeline(md.build_logistic_regression(), NUM, CAT, sampler=None)
    X, y = _data()
    pipe.fit(X, y)
    with pytest.raises(AttributeError):
        xai.builtin_importance(pipe)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def test_shap_explanation_shape_and_names(fitted, explanation):
    pipe, X, _ = fitted
    assert explanation.values.shape[0] == len(X)
    assert "num1" in list(explanation.feature_names)


def test_shap_values_sum_to_model_margin(fitted, explanation):
    """The additivity property — the guarantee that makes SHAP auditable."""
    pipe, X, _ = fitted
    frame = xai.preprocessed_frame(pipe, X)
    margin = pipe.named_steps["clf"].predict(frame.to_numpy(), output_margin=True)
    reconstructed = explanation.values.sum(axis=1) + explanation.base_values
    np.testing.assert_allclose(reconstructed, margin, rtol=1e-4, atol=1e-4)


def test_shap_explanation_carries_display_data(fitted, explanation):
    _, X, _ = fitted
    assert explanation.display_data is not None
    names = list(explanation.feature_names)
    col = np.asarray(explanation.display_data)[:, names.index("num2")]
    assert col.mean() == pytest.approx(X["num2"].mean())


def test_shap_sample_size_limits_rows(fitted):
    pytest.importorskip("shap")
    pipe, X, _ = fitted
    expl = xai.shap_explanation(pipe, X, sample_size=50)
    assert expl.values.shape[0] == 50


def test_shap_importance_sorted_descending(explanation):
    imp = xai.shap_importance(explanation)
    assert imp.is_monotonic_decreasing
    assert (imp >= 0).all()  # mean |SHAP| cannot be negative
    assert imp.index[0] == "num1"


def test_signed_direction_pushes_signal_the_right_way(fitted, explanation):
    """num1 is elevated in the minority class, so high num1 must mean fraud."""
    _, X, y = fitted
    values = explanation.values
    names = list(explanation.feature_names)
    shap_num1 = values[:, names.index("num1")]
    assert shap_num1[y.to_numpy() == 1].mean() > shap_num1[y.to_numpy() == 0].mean()


def test_importance_comparison_reports_rank_gap(fitted, explanation):
    pipe, _, _ = fitted
    comp = xai.importance_comparison(
        xai.builtin_importance(pipe), xai.shap_importance(explanation), top_n=3
    )
    assert list(comp.columns) == [
        "gain", "mean_abs_shap", "gain_rank", "shap_rank", "rank_gap",
    ]
    assert len(comp) == 3
    assert (comp["rank_gap"] == comp["gain_rank"] - comp["shap_rank"]).all()


def test_importance_comparison_caps_at_available_features(fitted, explanation):
    """top_n larger than the feature count must not error or pad."""
    pipe, _, _ = fitted
    n_features = len(explanation.feature_names)
    comp = xai.importance_comparison(
        xai.builtin_importance(pipe), xai.shap_importance(explanation), top_n=99
    )
    assert len(comp) == n_features


def test_rank_correlation_is_one_for_identical_rankings():
    s = pd.Series({"a": 3.0, "b": 2.0, "c": 1.0})
    assert xai.rank_correlation(s, s) == pytest.approx(1.0)


def test_rank_correlation_is_minus_one_for_reversed_rankings():
    ascending = pd.Series({"a": 3.0, "b": 2.0, "c": 1.0})
    descending = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    assert xai.rank_correlation(ascending, descending) == pytest.approx(-1.0)


def test_rank_correlation_needs_enough_overlap():
    a = pd.Series({"x": 1.0, "y": 2.0})
    b = pd.Series({"p": 1.0, "q": 2.0})
    assert np.isnan(xai.rank_correlation(a, b))


# --------------------------------------------------------------------------- #
# Case selection & local explanations
# --------------------------------------------------------------------------- #
def test_select_cases_picks_one_of_each_outcome():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_score = np.array([0.9, 0.2, 0.8, 0.1, 0.95, 0.05])
    cases = xai.select_cases(y_true, y_score, threshold=0.5)
    assert cases["true_positive"] == 4       # highest-scoring actual fraud
    assert cases["false_positive"] == 2      # highest-scoring legit above threshold
    assert cases["false_negative"] == 1      # lowest-scoring missed fraud
    assert cases["true_negative"] == 5       # lowest-scoring legit


def test_select_cases_omits_empty_groups():
    """A model with zero false positives must not fabricate one."""
    y_true = np.array([1, 0, 0])
    y_score = np.array([0.9, 0.1, 0.2])
    cases = xai.select_cases(y_true, y_score, threshold=0.5)
    assert "false_positive" not in cases
    assert "false_negative" not in cases
    assert set(cases) == {"true_positive", "true_negative"}


def test_case_table_reports_actual_and_prediction():
    y_true = np.array([1, 0])
    y_score = np.array([0.9, 0.1])
    cases = xai.select_cases(y_true, y_score, 0.5)
    table = xai.case_table(y_true, y_score, 0.5, cases)
    tp = table[table["case"] == "true_positive"].iloc[0]
    assert tp["actual"] == 1 and tp["predicted"] == 1


def test_top_contributors_ordered_by_magnitude(explanation):
    contrib = xai.top_contributors(explanation, row=0, n=3)
    assert len(contrib) == 3
    mags = contrib["shap"].abs().to_numpy()
    assert (np.diff(mags) <= 1e-12).all()
    assert set(contrib["direction"]) <= {"toward fraud", "toward legit"}


def test_top_contributors_uses_display_values(fitted, explanation):
    _, X, _ = fitted
    contrib = xai.top_contributors(explanation, row=0, n=len(explanation.feature_names))
    num2 = contrib[contrib["feature"] == "num2"].iloc[0]["value"]
    assert num2 == pytest.approx(X["num2"].iloc[0])


# --------------------------------------------------------------------------- #
# Effect shape
# --------------------------------------------------------------------------- #
def test_effect_shape_groups_by_raw_value(explanation):
    shape = xai.effect_shape(explanation, "num1", bins=[-10, 0, 10])
    assert list(shape.columns) == ["n", "mean_shap", "direction"]
    assert shape["n"].sum() == explanation.values.shape[0]
    # num1 > 0 is the fraud-elevated side.
    assert shape["mean_shap"].iloc[-1] > shape["mean_shap"].iloc[0]


def test_effect_shape_unknown_feature_raises(explanation):
    with pytest.raises(KeyError):
        xai.effect_shape(explanation, "nope")


# --------------------------------------------------------------------------- #
# Rule scoring
# --------------------------------------------------------------------------- #
def test_rule_performance_computes_recall_and_precision():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    perfect = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    loose = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    out = xai.rule_performance(y, {"perfect": perfect, "loose": loose}).set_index("rule")
    assert out.loc["perfect", "recall"] == pytest.approx(1.0)
    assert out.loc["perfect", "precision"] == pytest.approx(1.0)
    assert out.loc["perfect", "legit_flagged"] == 0
    assert out.loc["loose", "recall"] == pytest.approx(1.0)
    assert out.loc["loose", "precision"] == pytest.approx(0.6)
    assert out.loc["loose", "legit_flagged"] == 2


def test_rule_performance_handles_rule_that_flags_nothing():
    y = np.array([1, 0, 0])
    out = xai.rule_performance(y, {"never": np.zeros(3, dtype=bool)}).iloc[0]
    assert out["precision"] == 0.0 and out["recall"] == 0.0 and out["flagged"] == 0


def test_rule_performance_accepts_pandas_masks():
    y = pd.Series([1, 1, 0, 0])
    mask = pd.Series([True, False, False, False])
    out = xai.rule_performance(y, {"r": mask}).iloc[0]
    assert out["frauds_caught"] == 1 and out["recall"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_plot_builtin_importance_writes_file(fitted, tmp_path):
    pipe, _, _ = fitted
    path = tmp_path / "imp.png"
    xai.plot_builtin_importance(xai.builtin_importance(pipe), 5, "t", path)
    assert path.stat().st_size > 0


def test_plot_importance_comparison_writes_file(fitted, explanation, tmp_path):
    pipe, _, _ = fitted
    comp = xai.importance_comparison(
        xai.builtin_importance(pipe), xai.shap_importance(explanation), 5
    )
    path = tmp_path / "cmp.png"
    xai.plot_importance_comparison(comp, "t", path)
    assert path.stat().st_size > 0


def test_plot_shap_summary_and_bar_write_files(explanation, tmp_path):
    summary, bar = tmp_path / "s.png", tmp_path / "b.png"
    xai.plot_shap_summary(explanation, 5, "t", summary)
    xai.plot_shap_bar(explanation, 5, "t", bar)
    assert summary.stat().st_size > 0 and bar.stat().st_size > 0


def test_plot_force_and_waterfall_write_files(explanation, tmp_path):
    force, water = tmp_path / "f.png", tmp_path / "w.png"
    xai.plot_force(explanation, 0, "t", force)
    xai.plot_waterfall(explanation, 0, 5, "t", water)
    assert force.stat().st_size > 0 and water.stat().st_size > 0


def test_plot_dependence_writes_file(explanation, tmp_path):
    path = tmp_path / "d.png"
    xai.plot_dependence(explanation, "num1", "t", path)
    assert path.stat().st_size > 0


def test_plot_dependence_unknown_feature_raises(explanation):
    with pytest.raises(KeyError):
        xai.plot_dependence(explanation, "nope")
