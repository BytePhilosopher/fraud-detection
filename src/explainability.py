"""Model explainability for Task 3 — built-in importance and SHAP.

Why two importance measures rather than one
-------------------------------------------
XGBoost's built-in ``feature_importances_`` (gain) answers *"which features did
the trees split on most profitably?"* — a property of the fitted structure. It is
global, unsigned, and says nothing about direction: a high-gain feature could
push scores up, down, or both depending on its value.

SHAP answers *"how much did this feature move **this** prediction, and which
way?"* Values are additive per prediction (they sum to the model output minus the
base value), so they support both a global ranking (mean |SHAP|) and per-
transaction explanations — which is what a fraud analyst reviewing a flagged
transaction actually needs.

Reporting both is the check: where the two rankings disagree, the disagreement is
itself the finding (a feature can be split on constantly yet barely move the
output, or move it hard in a small, decisive minority of cases).

Everything here operates on the **preprocessed** matrix, because the fitted
estimator lives behind a ``ColumnTransformer``. Feature names are recovered from
the transformer so plots stay readable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE


# --------------------------------------------------------------------------- #
# Bridging the pipeline: model-space features
# --------------------------------------------------------------------------- #
def preprocessed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted preprocessor and return a named DataFrame.

    SHAP and the built-in importances both live in the estimator's feature space
    (post-scaling, post-one-hot), so every explanation has to be computed there.
    Recovering ``get_feature_names_out()`` keeps that space interpretable.
    """
    pre = pipeline.named_steps["preprocess"]
    arr = pre.transform(X)
    return pd.DataFrame(arr, columns=pre.get_feature_names_out(), index=X.index)


def tidy_names(names) -> list[str]:
    """Strip ColumnTransformer prefixes (``num__``, ``cat__``) for display."""
    return [str(n).split("__", 1)[-1] for n in names]


def display_frame(pipeline, X: pd.DataFrame, model_frame: pd.DataFrame) -> pd.DataFrame:
    """``model_frame`` with scaled numerics swapped back to original units.

    SHAP has to be computed on the standardised matrix, but a plot labelled
    ``time_since_signup = -1.579`` tells a fraud analyst nothing. The numeric
    block maps 1:1 back to the input columns, so the raw values can be
    substituted for display without touching a single attribution. One-hot
    columns keep their 0/1 indicator, which is already readable.
    """
    out = model_frame.copy()
    # model_frame may still carry ColumnTransformer prefixes (num__/cat__), so
    # match on the tidied name rather than the literal column label.
    by_tidy = dict(zip(tidy_names(out.columns), out.columns))
    for col in X.columns:
        target = by_tidy.get(col)
        if target is not None and pd.api.types.is_numeric_dtype(X[col]):
            out[target] = X.loc[model_frame.index, col].to_numpy()
    return out


# --------------------------------------------------------------------------- #
# Built-in (gain) importance
# --------------------------------------------------------------------------- #
def builtin_importance(
    pipeline, importance_type: str = "gain", tidy: bool = True
) -> pd.Series:
    """Tree-model feature importance, descending.

    ``gain`` (the XGBoost default for ``feature_importances_``) is the average
    improvement in the split criterion each feature delivered. ``weight`` counts
    raw split occurrences and is biased toward high-cardinality numerics, which
    is why gain is the default here.
    """
    clf = pipeline.named_steps["clf"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()

    if importance_type == "gain" and hasattr(clf, "feature_importances_"):
        values = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "get_booster"):
        # Booster keys are f0..fN and omit features never used in a split.
        scores = clf.get_booster().get_score(importance_type=importance_type)
        values = np.array(
            [scores.get(f"f{i}", 0.0) for i in range(len(names))], dtype=float
        )
        total = values.sum()
        if total > 0:
            values = values / total
    else:
        raise AttributeError(
            f"{type(clf).__name__} exposes no feature importances"
        )

    index = tidy_names(names) if tidy else list(names)
    return pd.Series(values, index=index).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def shap_explanation(
    pipeline,
    X: pd.DataFrame,
    sample_size: int | None = None,
    random_state: int = RANDOM_STATE,
    tidy: bool = True,
):
    """SHAP values for ``X`` under the pipeline's tree estimator.

    ``TreeExplainer`` is exact for tree ensembles (no kernel approximation), so
    the only reason to sample is plot legibility and runtime on large frames.
    Sampling is stratified by nothing — callers who need specific cases should
    explain those rows directly (see :func:`select_cases`).

    Returns a ``shap.Explanation`` whose ``.values`` are in **log-odds** units:
    additive on the raw margin, not on probability. ``display_data`` carries the
    unscaled feature values so plots label points in business units.
    """
    import shap

    frame = preprocessed_frame(pipeline, X)
    if sample_size is not None and sample_size < len(frame):
        frame = frame.sample(sample_size, random_state=random_state)
    display = display_frame(pipeline, X, frame)
    if tidy:
        frame.columns = tidy_names(frame.columns)
        display.columns = tidy_names(display.columns)

    explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
    explanation = explainer(frame)
    explanation.display_data = display.to_numpy()
    return explanation


def shap_importance(explanation) -> pd.Series:
    """Global importance as mean |SHAP| per feature, descending.

    The mean absolute value is the standard global summary: it measures average
    magnitude of impact on the output, ignoring direction.
    """
    values = np.abs(explanation.values)
    if values.ndim == 3:  # (n, features, classes) — take the positive class
        values = values[:, :, -1]
    return pd.Series(
        values.mean(axis=0), index=list(explanation.feature_names)
    ).sort_values(ascending=False)


def signed_shap_direction(explanation) -> pd.Series:
    """Mean signed SHAP per feature — which way each feature pushes on average.

    Positive means the feature pushes predictions toward fraud on average. Read
    alongside mean |SHAP|: a feature can have large magnitude and a near-zero
    mean when it pushes both ways depending on its value.
    """
    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, -1]
    return pd.Series(values.mean(axis=0), index=list(explanation.feature_names))


def importance_comparison(
    builtin: pd.Series, shap_imp: pd.Series, top_n: int = 15
) -> pd.DataFrame:
    """Side-by-side gain vs mean|SHAP| ranking, with the rank gap.

    ``rank_gap`` = builtin rank - SHAP rank. A large positive gap means SHAP
    considers the feature more important than gain does.
    """
    union = list(dict.fromkeys(list(builtin.index) + list(shap_imp.index)))
    df = pd.DataFrame(index=union)
    df["gain"] = builtin.reindex(union).fillna(0.0)
    df["mean_abs_shap"] = shap_imp.reindex(union).fillna(0.0)
    df["gain_rank"] = df["gain"].rank(ascending=False, method="min").astype(int)
    df["shap_rank"] = df["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    df["rank_gap"] = df["gain_rank"] - df["shap_rank"]
    return df.sort_values("mean_abs_shap", ascending=False).head(top_n)


def rank_correlation(builtin: pd.Series, shap_imp: pd.Series) -> float:
    """Spearman correlation between the two importance rankings."""
    common = [f for f in builtin.index if f in shap_imp.index]
    if len(common) < 3:
        return float("nan")
    return float(
        pd.Series(builtin[common]).corr(pd.Series(shap_imp[common]), method="spearman")
    )


# --------------------------------------------------------------------------- #
# Case selection for local explanations
# --------------------------------------------------------------------------- #
def select_cases(y_true, y_score, threshold: float) -> dict[str, int]:
    """Pick the most instructive TP / FP / FN / TN row for local explanation.

    Extremes are chosen deliberately rather than random members of each group:

    * ``true_positive``  — highest-scoring caught fraud: the model's clearest win.
    * ``false_positive`` — highest-scoring legitimate row: the *worst* false
      alarm, so its SHAP breakdown shows which features misfire hardest.
    * ``false_negative`` — lowest-scoring missed fraud: the hardest miss, showing
      which evidence was absent.
    * ``true_negative``  — lowest-scoring legitimate row, as a contrast baseline.

    Returns positional indices into ``y_true``/``y_score``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    pred = y_score >= threshold

    groups = {
        "true_positive": np.flatnonzero((y_true == 1) & pred),
        "false_positive": np.flatnonzero((y_true == 0) & pred),
        "false_negative": np.flatnonzero((y_true == 1) & ~pred),
        "true_negative": np.flatnonzero((y_true == 0) & ~pred),
    }
    picks: dict[str, int] = {}
    for name, idx in groups.items():
        if len(idx) == 0:
            continue  # e.g. a model with zero false positives at this threshold
        # Highest score for the "confidently flagged" groups, lowest otherwise.
        chooser = np.argmax if name in ("true_positive", "false_positive") else np.argmin
        picks[name] = int(idx[chooser(y_score[idx])])
    return picks


def case_table(y_true, y_score, threshold: float, cases: dict[str, int]) -> pd.DataFrame:
    """Summarise the selected cases: actual label, score, prediction."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    rows = [
        {
            "case": name,
            "row": pos,
            "actual": int(y_true[pos]),
            "fraud_score": float(y_score[pos]),
            "predicted": int(y_score[pos] >= threshold),
        }
        for name, pos in cases.items()
    ]
    return pd.DataFrame(rows)


def top_contributors(explanation, row: int, n: int = 6) -> pd.DataFrame:
    """The n largest-magnitude SHAP contributions for a single prediction."""
    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, -1]
    # Prefer unscaled values so the table reads in business units.
    data = getattr(explanation, "display_data", None)
    if data is None:
        data = explanation.data
    contrib = pd.DataFrame(
        {
            "feature": list(explanation.feature_names),
            "value": np.asarray(data)[row],
            "shap": values[row],
        }
    )
    contrib["direction"] = np.where(contrib["shap"] > 0, "toward fraud", "toward legit")
    order = contrib["shap"].abs().sort_values(ascending=False).index
    return contrib.loc[order].head(n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_builtin_importance(importance: pd.Series, top_n: int = 10, title="", path=None):
    """Horizontal bar chart of the top-n built-in importances."""
    import matplotlib.pyplot as plt

    top = importance.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(top) + 1.5))
    ax.barh(top.index, top.to_numpy(), color="steelblue")
    ax.set(xlabel="importance (gain, normalised)", title=title)
    for y, v in enumerate(top.to_numpy()):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig, ax


def plot_importance_comparison(comparison: pd.DataFrame, title="", path=None):
    """Gain vs mean|SHAP| as paired bars, both min-max scaled to [0, 1].

    Scaling is required because gain is a normalised share while mean |SHAP| is
    in log-odds; only the *ordering* is comparable.
    """
    import matplotlib.pyplot as plt

    def scale(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0.0

    df = comparison.iloc[::-1]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(df) + 1.8))
    ax.barh(y - 0.2, scale(df["gain"]), height=0.4, label="gain (built-in)",
            color="steelblue")
    ax.barh(y + 0.2, scale(df["mean_abs_shap"]), height=0.4, label="mean |SHAP|",
            color="darkorange")
    ax.set(yticks=y, yticklabels=df.index, xlabel="relative importance (scaled)",
           title=title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig, ax


def plot_shap_summary(explanation, top_n: int = 12, title="", path=None):
    """SHAP beeswarm — global importance *and* direction in one figure.

    Each dot is one transaction. Position on x is that feature's SHAP value
    (right = pushed toward fraud); colour is the feature's own value. This is the
    plot that shows whether *high* or *low* values drive fraud.
    """
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=top_n, show=False)
    fig = plt.gcf()
    if title:
        fig.axes[0].set_title(title, fontsize=11)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_shap_bar(explanation, top_n: int = 12, title="", path=None):
    """SHAP global bar chart (mean |SHAP| per feature)."""
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.plots.bar(explanation, max_display=top_n, show=False)
    fig = plt.gcf()
    if title:
        fig.axes[0].set_title(title, fontsize=11)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_force(explanation, row: int, title="", path=None):
    """Static SHAP force plot for one prediction.

    ``matplotlib=True`` renders without the JavaScript runtime, so the figure
    survives being saved to a file and viewed outside a notebook.
    """
    import matplotlib.pyplot as plt
    import shap

    single = explanation[row]
    display = getattr(explanation, "display_data", None)
    features = np.asarray(display)[row] if display is not None else single.data
    shap.plots.force(
        single.base_values,
        single.values,
        features=features,
        feature_names=list(explanation.feature_names),
        matplotlib=True,
        show=False,
    )
    fig = plt.gcf()
    if title:
        fig.suptitle(title, fontsize=11, y=1.05)
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_waterfall(explanation, row: int, top_n: int = 10, title="", path=None):
    """Waterfall for one prediction — the readable sibling of the force plot.

    Shows how each feature moves the log-odds from the base value to this
    prediction, largest contribution first.
    """
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.plots.waterfall(explanation[row], max_display=top_n, show=False)
    fig = plt.gcf()
    if title:
        fig.axes[0].set_title(title, fontsize=11)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def effect_shape(explanation, feature: str, bins=None) -> pd.DataFrame:
    """Mean SHAP for ``feature`` grouped by its own (unscaled) value.

    A global bar chart says *how much* a feature matters; this says *where* the
    effect switches sign. Discrete features are grouped by exact value; for
    continuous ones pass explicit ``bins``. This is the table that turns a SHAP
    ranking into a threshold a rules engine can act on.
    """
    names = list(explanation.feature_names)
    if feature not in names:
        raise KeyError(f"{feature!r} not in explanation features")
    col = names.index(feature)
    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, -1]
    data = getattr(explanation, "display_data", None)
    if data is None:
        data = explanation.data
    raw = np.asarray(data)[:, col]

    key = pd.cut(raw, bins=bins) if bins is not None else pd.Series(raw)
    grouped = pd.DataFrame({"raw": raw, "shap": values[:, col]}).groupby(
        key, observed=True
    )
    out = grouped.agg(n=("shap", "size"), mean_shap=("shap", "mean"))
    out["direction"] = np.where(out["mean_shap"] > 0, "toward fraud", "toward legit")
    out.index.name = feature
    return out


def rule_performance(y_true, rules: dict[str, "pd.Series | np.ndarray"]) -> pd.DataFrame:
    """Score candidate hard rules against the labels.

    A SHAP threshold is only a recommendation once someone knows what enforcing
    it would cost. Each rule is a boolean mask over the same rows as ``y_true``;
    the output gives what it would catch (``recall``), how clean the flags are
    (``precision``), and how many legitimate customers it would inconvenience.
    """
    y = np.asarray(y_true).astype(bool)
    total_fraud = int(y.sum())
    rows = []
    for name, mask in rules.items():
        m = np.asarray(mask).astype(bool)
        flagged = int(m.sum())
        caught = int((m & y).sum())
        rows.append(
            {
                "rule": name,
                "flagged": flagged,
                "frauds_caught": caught,
                "legit_flagged": flagged - caught,
                "recall": caught / total_fraud if total_fraud else float("nan"),
                "precision": caught / flagged if flagged else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_dependence(explanation, feature: str, title="", path=None, max_points=5000):
    """SHAP value against the feature's own unscaled value.

    Reveals thresholds a global bar chart hides — e.g. ``time_since_signup``
    flipping from strongly-fraud to mildly-legitimate within the first hour.
    Hand-plotted rather than delegated to ``shap.plots.scatter`` so the x-axis
    stays in business units instead of standard deviations.
    """
    import matplotlib.pyplot as plt

    names = list(explanation.feature_names)
    if feature not in names:
        raise KeyError(f"{feature!r} not in explanation features")
    col = names.index(feature)
    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, -1]
    data = getattr(explanation, "display_data", None)
    if data is None:
        data = explanation.data
    x = np.asarray(data)[:, col].astype(float)
    y = values[:, col]
    if len(x) > max_points:
        rng = np.random.RandomState(RANDOM_STATE)
        sel = rng.choice(len(x), max_points, replace=False)
        x, y = x[sel], y[sel]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, s=6, alpha=0.3, c=np.where(y > 0, "firebrick", "steelblue"))
    ax.axhline(0, ls="--", c="grey", lw=1)
    ax.set(xlabel=f"{feature} (original units)",
           ylabel="SHAP value (log-odds)",
           title=title or f"SHAP dependence: {feature}")
    ax.text(0.99, 0.95, "above 0 = pushes toward fraud", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="firebrick")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig, ax
