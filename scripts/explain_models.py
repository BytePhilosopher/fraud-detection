"""Task 3 — explain the selected models with built-in importance and SHAP.

For each dataset this reproduces the Task-2 split (same `random_state`, so the
test rows are identical), loads the selected pipeline, then:

    1. extracts built-in gain importance and plots the top 10
    2. computes exact TreeExplainer SHAP values on the test set
    3. emits the global summary (beeswarm + bar) and a gain-vs-SHAP comparison
    4. picks one TP / FP / FN / TN and emits force + waterfall plots for each
    5. writes every ranking and local breakdown to CSV

Usage
-----
    python scripts/explain_models.py                  # both datasets
    python scripts/explain_models.py --dataset fraud
    python scripts/explain_models.py --sample 5000    # subsample the beeswarm

Artifacts land in ``reports/figures/`` and ``reports/``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src import (  # noqa: E402
    config,
    evaluation as ev,
    explainability as xai,
    modeling as md,
)

#: Beeswarm legibility, not runtime, is the binding constraint — TreeExplainer is
#: exact and fast. Rankings are computed on the FULL test set regardless.
PLOT_SAMPLE = 5000

#: Business-meaningful buckets for continuous features whose interesting
#: structure sits at one end of the range. `time_since_signup` is the case that
#: matters: 95% of rows are >24h, so quantile bins would smear the first hour —
#: exactly where the signal lives — across a single bucket.
EFFECT_BINS = {
    "time_since_signup": [-1, 1, 6, 24, 168, 720, 1440, 3000],  # h: 1h/6h/1d/1w/1m/2m
}


def effect_bins(explanation, feature: str):
    """Bucket edges for :func:`xai.effect_shape`, or None to group exact values."""
    if feature in EFFECT_BINS:
        return EFFECT_BINS[feature]
    names = list(explanation.feature_names)
    data = getattr(explanation, "display_data", None)
    if data is None:
        data = explanation.data
    raw = pd.Series(pd.array(data[:, names.index(feature)]))
    if raw.nunique() <= 20:
        return None  # discrete: exact values are more informative than bins
    return list(raw.quantile([0, .1, .25, .5, .75, .9, 1.0]).unique())


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_dataset(dataset: str, features: list[str]):
    """Rebuild the exact Task-2 test split for a dataset."""
    if dataset == "fraud":
        df = pd.read_csv(config.FRAUD_FEATURES)
        X, y = md.split_features_target(df, config.FRAUD_TARGET, features)
        for c in md.FRAUD_CATEGORICAL_COLS:
            X[c] = X[c].astype(str)
    else:
        df = pd.read_csv(config.CREDITCARD_CLEAN)
        X, y = md.split_features_target(df, config.CREDITCARD_TARGET, features)
    return md.stratified_split(X, y)


def run_dataset(dataset: str, plot_sample: int = PLOT_SAMPLE) -> None:
    artifact = joblib.load(config.MODELS_DIR / f"{dataset}_selected.joblib")
    pipeline, threshold = artifact["pipeline"], artifact["threshold"]
    log(f"{dataset}: loaded {artifact.get('model_key', 'selected')} "
        f"(threshold {threshold:.4f})")

    features = artifact.get("features") or list(
        pipeline.named_steps["preprocess"].feature_names_in_
    )
    _, X_test, _, y_test = load_dataset(dataset, features)
    scores = ev.predict_proba_positive(pipeline, X_test)
    log(f"{dataset}: explaining {len(X_test):,} test rows "
        f"({int(y_test.sum())} frauds)")

    fig_dir, rep_dir = config.FIGURES_DIR, config.REPORTS_DIR

    # ---- 1. built-in importance ------------------------------------------- #
    gain = xai.builtin_importance(pipeline, importance_type="gain")
    weight = xai.builtin_importance(pipeline, importance_type="weight")
    gain.rename("gain").to_frame().join(weight.rename("weight")).to_csv(
        rep_dir / f"task3_builtin_importance_{dataset}.csv"
    )
    xai.plot_builtin_importance(
        gain, 10,
        f"{dataset} — XGBoost built-in importance (gain), top 10",
        fig_dir / f"{dataset}_builtin_importance.png",
    )
    plt.close("all")
    log(f"{dataset}: top-10 gain\n{gain.head(10).round(4).to_string()}")

    # ---- 2. SHAP ---------------------------------------------------------- #
    t0 = time.time()
    full = xai.shap_explanation(pipeline, X_test)  # rankings on every test row
    log(f"{dataset}: SHAP values for {full.values.shape[0]:,} rows in "
        f"{time.time() - t0:.0f}s")

    shap_imp = xai.shap_importance(full)
    signed = xai.signed_shap_direction(full)
    shap_imp.rename("mean_abs_shap").to_frame().join(
        signed.rename("mean_signed_shap")
    ).to_csv(rep_dir / f"task3_shap_importance_{dataset}.csv")
    log(f"{dataset}: top-10 mean|SHAP|\n{shap_imp.head(10).round(4).to_string()}")

    # Beeswarm on a subsample: 30k overlapping dots is unreadable, and the
    # ranking it displays is unchanged.
    plot_expl = full
    if plot_sample and plot_sample < full.values.shape[0]:
        plot_expl = xai.shap_explanation(pipeline, X_test, sample_size=plot_sample)
    xai.plot_shap_summary(
        plot_expl, 12,
        f"{dataset} — SHAP summary (beeswarm, {plot_expl.values.shape[0]:,} rows)",
        fig_dir / f"{dataset}_shap_summary.png",
    )
    plt.close("all")
    xai.plot_shap_bar(
        plot_expl, 12,
        f"{dataset} — SHAP global importance (mean |SHAP|)",
        fig_dir / f"{dataset}_shap_bar.png",
    )
    plt.close("all")

    # ---- 3. gain vs SHAP -------------------------------------------------- #
    comparison = xai.importance_comparison(gain, shap_imp, top_n=15)
    comparison.to_csv(rep_dir / f"task3_importance_comparison_{dataset}.csv")
    rho = xai.rank_correlation(gain, shap_imp)
    xai.plot_importance_comparison(
        comparison,
        f"{dataset} — gain vs mean|SHAP| (Spearman rho = {rho:.2f})",
        fig_dir / f"{dataset}_importance_comparison.png",
    )
    plt.close("all")
    log(f"{dataset}: Spearman rho(gain, mean|SHAP|) = {rho:.4f}\n"
        f"{comparison.round(4).to_string()}")

    # ---- 4. local explanations: TP / FP / FN / TN -------------------------- #
    cases = xai.select_cases(y_test, scores, threshold)
    table = xai.case_table(y_test, scores, threshold, cases)
    table.to_csv(rep_dir / f"task3_cases_{dataset}.csv", index=False)
    log(f"{dataset}: selected cases\n{table.to_string(index=False)}")

    raw = X_test.reset_index(drop=True)
    local_rows = []
    for name, pos in cases.items():
        pretty = name.replace("_", " ")
        score = float(scores[pos])
        title = f"{dataset} — {pretty} (fraud score {score:.4f})"
        xai.plot_force(full, pos, f"Force plot: {title}",
                       fig_dir / f"{dataset}_force_{name}.png")
        plt.close("all")
        xai.plot_waterfall(full, pos, 10, f"Waterfall: {title}",
                           fig_dir / f"{dataset}_waterfall_{name}.png")
        plt.close("all")

        contrib = xai.top_contributors(full, pos, 6)
        log(f"{dataset}/{name} (score {score:.4f}) top contributions\n"
            f"{contrib.round(4).to_string(index=False)}")
        log(f"{dataset}/{name} raw feature values\n"
            f"{raw.loc[pos].to_dict()}")
        local_rows.append(contrib.assign(case=name, fraud_score=score))

    pd.concat(local_rows, ignore_index=True).to_csv(
        rep_dir / f"task3_local_contributions_{dataset}.csv", index=False
    )

    # ---- 5. effect shape: where each top feature flips sign ---------------- #
    # This is what turns a SHAP ranking into a number a rules engine can use.
    shapes = []
    for feature in shap_imp.head(3).index:
        try:
            xai.plot_dependence(
                plot_expl, feature,
                title=f"{dataset} — SHAP dependence: {feature}",
                path=fig_dir / f"{dataset}_dependence_{feature}.png",
            )
            shape = xai.effect_shape(full, feature, bins=effect_bins(full, feature))
            log(f"{dataset}: SHAP effect of {feature} by value\n"
                f"{shape.round(4).to_string()}")
            shapes.append(shape.reset_index().rename(columns={feature: "bucket"})
                          .assign(feature=feature))
        except (KeyError, ValueError) as exc:
            log(f"{dataset}: skipped effect shape for {feature} ({exc})")
        plt.close("all")
    if shapes:
        pd.concat(shapes, ignore_index=True).to_csv(
            rep_dir / f"task3_effect_shape_{dataset}.csv", index=False
        )

    log(f"{dataset}: done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["fraud", "creditcard", "both"],
                    default="both")
    ap.add_argument("--sample", type=int, default=PLOT_SAMPLE,
                    help="rows in the beeswarm (0 = all); rankings always use all")
    args = ap.parse_args()

    config.ensure_dirs()
    names = ["fraud", "creditcard"] if args.dataset == "both" else [args.dataset]
    for name in names:
        run_dataset(name, plot_sample=args.sample)


if __name__ == "__main__":
    main()
