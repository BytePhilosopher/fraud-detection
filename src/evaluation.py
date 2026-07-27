"""Evaluation for imbalanced fraud classification.

Why these metrics
-----------------
Accuracy is excluded on purpose: a constant "legitimate" prediction scores
90.6% on Fraud_Data and 99.83% on the credit-card set while catching zero fraud.

* **AUC-PR (average precision)** — primary. Summarises the precision/recall
  trade-off across every threshold and, unlike ROC-AUC, its baseline is the
  positive rate, so it stays sensitive when positives are 0.17% of the data.
* **F1** — single-threshold balance of precision and recall.
* **Precision / recall** — the two business quantities: recall is fraud caught,
  precision is how many blocked transactions were actually fraud.
* **Confusion matrix** — the raw counts. False negatives are fraud losses; false
  positives are blocked legitimate customers.
* **ROC-AUC** — reported for continuity with the literature, read with caution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from .config import RANDOM_STATE

#: Scorers used for cross-validation. AUC-PR first — it is the selection metric.
CV_SCORING = {
    "auc_pr": "average_precision",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


def predict_proba_positive(estimator, X) -> np.ndarray:
    """Positive-class scores, whatever the estimator exposes."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    raise AttributeError("estimator exposes neither predict_proba nor decision_function")


def classification_metrics(
    y_true, y_score, threshold: float = 0.5, label: str | None = None
) -> dict:
    """Threshold-free (AUC-PR, ROC-AUC) and thresholded (F1, P, R) metrics.

    ``y_score`` must be continuous positive-class scores, not hard labels —
    AUC-PR is undefined on binarised predictions.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "auc_pr": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if label is not None:
        out = {"model": label, **out}
    return out


def confusion_frame(y_true, y_score, threshold: float = 0.5) -> pd.DataFrame:
    """Confusion matrix with business-readable labels."""
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["actual: legitimate", "actual: fraud"],
        columns=["predicted: legitimate", "predicted: fraud"],
    )


def best_f1_threshold(y_true, y_score) -> tuple[float, float]:
    """Threshold maximising F1 on the given scores, and that F1.

    Used to pick an operating point on *training* cross-validated predictions —
    never on the test set, which would make the reported F1 an oracle value.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1.
    p, r = precision[:-1], recall[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    if len(f1) == 0:
        return 0.5, 0.0
    i = int(np.nanargmax(f1))
    return float(thresholds[i]), float(f1[i])


def select_threshold(estimator, X, y, cv: int = 3) -> tuple[float, float]:
    """Choose a decision threshold using out-of-fold training predictions.

    The default 0.5 cut-off is arbitrary for a resampled model — SMOTE shifts the
    score distribution, so 0.5 no longer corresponds to the prior. This refits
    the pipeline across ``cv`` folds of the training data, collects out-of-fold
    probabilities, and returns the F1-maximising threshold. Honest, because no
    test data is involved.
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(estimator, X, y, cv=splitter, method="predict_proba")[:, 1]
    return best_f1_threshold(y, proba)


def cross_validate_model(
    estimator, X, y, cv: int = 5, label: str | None = None
) -> pd.DataFrame:
    """Stratified K-fold CV returning per-metric mean and std.

    ``estimator`` should be the *pipeline* (preprocessing + SMOTE + model) so
    every fold refits its own transformers and sampler — otherwise the folds
    share information and the estimates are optimistic.
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        estimator, X, y, cv=splitter, scoring=CV_SCORING, n_jobs=1, error_score="raise"
    )
    rows = {}
    for name in CV_SCORING:
        vals = scores[f"test_{name}"]
        rows[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    out = pd.DataFrame(rows).T
    out.index.name = "metric"
    if label is not None:
        out["model"] = label
    return out


def cv_summary_row(cv_frame: pd.DataFrame, label: str) -> dict:
    """Flatten a ``cross_validate_model`` frame into one 'mean ± std' row."""
    row = {"model": label}
    for metric in cv_frame.index:
        m, s = cv_frame.loc[metric, "mean"], cv_frame.loc[metric, "std"]
        row[metric] = f"{m:.4f} ± {s:.4f}"
    return row


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Side-by-side model comparison, ranked by AUC-PR where available."""
    df = pd.DataFrame(rows)
    if "auc_pr" in df.columns and pd.api.types.is_numeric_dtype(df["auc_pr"]):
        df = df.sort_values("auc_pr", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_pr_curves(curves: dict[str, tuple], title: str, path=None):
    """Precision-recall curves for several models on one axis.

    ``curves`` maps a label to ``(y_true, y_score)``. The dashed line is the
    no-skill baseline (= the positive rate), which is where AUC-PR starts from.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    baseline = None
    for label, (y_true, y_score) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, label=f"{label} (AUC-PR = {ap:.3f})")
        baseline = float(np.mean(np.asarray(y_true)))
    if baseline is not None:
        ax.axhline(
            baseline, ls="--", c="grey", lw=1, label=f"no-skill ({baseline:.4f})"
        )
    ax.set(xlabel="Recall", ylabel="Precision", title=title, ylim=(0, 1.02))
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig, ax


def plot_confusion_matrices(mats: dict[str, pd.DataFrame], title: str, path=None):
    """Annotated heatmaps of each model's confusion matrix."""
    import matplotlib.pyplot as plt

    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (label, cm) in zip(axes, mats.items()):
        ax.imshow(cm.to_numpy(), cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm.iat[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:,}",
                    ha="center",
                    va="center",
                    color="white" if v > cm.to_numpy().max() / 2 else "black",
                )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["pred legit", "pred fraud"],
            yticklabels=["legit", "fraud"],
            title=label,
        )
    fig.suptitle(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig, axes
