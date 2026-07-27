"""Tests for the imbalanced-classification metric helpers (Task 2)."""
import numpy as np
import pandas as pd
import pytest

from src import evaluation, modeling


def _scores():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.05, 0.10, 0.20, 0.60, 0.70, 0.90])
    return y_true, y_score


def test_classification_metrics_confusion_counts():
    y_true, y_score = _scores()
    m = evaluation.classification_metrics(y_true, y_score, threshold=0.5)
    # >=0.5 -> rows 3,4,5 predicted fraud: one FP, two TP.
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 0, 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(0.8)


def test_classification_metrics_perfect_ranking_gives_auc_pr_one():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluation.classification_metrics(y_true, y_score)
    assert m["auc_pr"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_classification_metrics_threshold_shifts_tradeoff():
    y_true, y_score = _scores()
    strict = evaluation.classification_metrics(y_true, y_score, threshold=0.65)
    assert strict["precision"] == pytest.approx(1.0)  # the FP is now excluded
    assert strict["fp"] == 0
    # AUC-PR is threshold-free, so it must not move.
    loose = evaluation.classification_metrics(y_true, y_score, threshold=0.5)
    assert strict["auc_pr"] == pytest.approx(loose["auc_pr"])


def test_classification_metrics_label_is_first_key():
    y_true, y_score = _scores()
    m = evaluation.classification_metrics(y_true, y_score, label="lr")
    assert next(iter(m)) == "model" and m["model"] == "lr"


def test_no_predicted_positives_does_not_divide_by_zero():
    y_true, y_score = _scores()
    m = evaluation.classification_metrics(y_true, y_score, threshold=0.99)
    assert m["precision"] == 0.0 and m["f1"] == 0.0 and m["tp"] == 0


def test_confusion_frame_is_labelled():
    y_true, y_score = _scores()
    cm = evaluation.confusion_frame(y_true, y_score, threshold=0.5)
    assert cm.loc["actual: fraud", "predicted: fraud"] == 2
    assert cm.loc["actual: legitimate", "predicted: fraud"] == 1


def test_best_f1_threshold_finds_separating_cut():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    thr, f1 = evaluation.best_f1_threshold(y_true, y_score)
    assert f1 == pytest.approx(1.0)
    assert 0.3 < thr <= 0.8


def test_predict_proba_positive_uses_decision_function_fallback():
    class _DF:
        def decision_function(self, X):
            return np.arange(len(X), dtype=float)

    out = evaluation.predict_proba_positive(_DF(), np.zeros((4, 2)))
    assert out.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_predict_proba_positive_raises_without_scores():
    class _Nothing:
        pass

    with pytest.raises(AttributeError):
        evaluation.predict_proba_positive(_Nothing(), np.zeros((2, 2)))


def _pipeline_data():
    rng = np.random.RandomState(1)
    X = pd.DataFrame({"num1": rng.randn(200), "num2": rng.randn(200)})
    y = pd.Series([0] * 180 + [1] * 20)
    X.loc[y == 1, "num1"] += 3.0
    return X, y


def test_cross_validate_model_reports_mean_and_std():
    X, y = _pipeline_data()
    pipe = modeling.build_pipeline(
        modeling.build_logistic_regression(), ["num1", "num2"], []
    )
    frame = evaluation.cross_validate_model(pipe, X, y, cv=3)
    assert set(frame.index) == set(evaluation.CV_SCORING)
    assert {"mean", "std"} <= set(frame.columns)
    assert 0.0 <= frame.loc["auc_pr", "mean"] <= 1.0
    assert frame.loc["auc_pr", "std"] >= 0.0


def test_cv_summary_row_formats_mean_plus_minus_std():
    X, y = _pipeline_data()
    pipe = modeling.build_pipeline(
        modeling.build_logistic_regression(), ["num1", "num2"], []
    )
    frame = evaluation.cross_validate_model(pipe, X, y, cv=3)
    row = evaluation.cv_summary_row(frame, "lr")
    assert row["model"] == "lr"
    assert "±" in row["auc_pr"]


def test_select_threshold_uses_out_of_fold_scores():
    X, y = _pipeline_data()
    pipe = modeling.build_pipeline(
        modeling.build_logistic_regression(), ["num1", "num2"], []
    )
    thr, f1 = evaluation.select_threshold(pipe, X, y, cv=3)
    assert 0.0 <= thr <= 1.0
    assert f1 > y.mean()  # must beat the no-skill F1 floor


def test_comparison_table_ranks_by_auc_pr():
    rows = [
        {"model": "a", "auc_pr": 0.5, "f1": 0.4},
        {"model": "b", "auc_pr": 0.8, "f1": 0.3},
    ]
    table = evaluation.comparison_table(rows)
    assert table.loc[0, "model"] == "b"


def test_plot_helpers_write_files(tmp_path):
    y_true, y_score = _scores()
    pr = tmp_path / "pr.png"
    evaluation.plot_pr_curves({"m": (y_true, y_score)}, "t", pr)
    cm = tmp_path / "cm.png"
    evaluation.plot_confusion_matrices(
        {"m": evaluation.confusion_frame(y_true, y_score)}, "t", cm
    )
    assert pr.exists() and pr.stat().st_size > 0
    assert cm.exists() and cm.stat().st_size > 0
