"""Task 2 — train, tune, cross-validate and compare fraud models.

Runs the identical protocol over both datasets:

    stratified 80/20 split
      -> grid-search each candidate on the TRAIN half (3-fold, scored by AUC-PR)
      -> 5-fold stratified CV of each tuned pipeline (mean +/- std)
      -> pick a decision threshold from out-of-fold TRAIN probabilities
      -> score once on the untouched TEST half

Candidates per dataset:
    logreg_smote   Logistic Regression + SMOTE          (interpretable baseline)
    xgb_smote      XGBoost + SMOTE                      (ensemble, resampling)
    xgb_weighted   XGBoost + scale_pos_weight           (ensemble, cost-weighting)

The third exists so the imbalance *strategy* is compared, not just the model
family: SMOTE synthesises minority rows, scale_pos_weight reweights them.

Usage
-----
    python scripts/train_models.py                 # both datasets
    python scripts/train_models.py --dataset fraud # one dataset
    python scripts/train_models.py --quick         # 10% sample, smoke test

Artifacts land in ``models/`` (fitted pipelines) and ``reports/`` (metric tables,
figures, JSON summary).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: this script only writes files

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, evaluation as ev, modeling as md  # noqa: E402

CV_FOLDS = 5
TUNE_FOLDS = 3
THRESHOLD_FOLDS = 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Dataset definitions
# --------------------------------------------------------------------------- #
def load_fraud() -> tuple[pd.DataFrame, pd.Series]:
    """Fraud_Data with the Task-1 engineered features (incl. geolocated country)."""
    path = config.FRAUD_FEATURES
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run notebooks/feature-engineering.ipynb first."
        )
    df = pd.read_csv(path)
    cols = md.FRAUD_NUMERIC_COLS + md.FRAUD_CATEGORICAL_COLS
    X, y = md.split_features_target(df, config.FRAUD_TARGET, cols)
    for c in md.FRAUD_CATEGORICAL_COLS:
        X[c] = X[c].astype(str)  # OneHotEncoder wants str, not pandas category
    return X, y


def load_creditcard() -> tuple[pd.DataFrame, pd.Series]:
    """Cleaned credit-card transactions (Time, V1-V28, Amount)."""
    path = config.CREDITCARD_CLEAN
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run notebooks/eda-creditcard.ipynb first."
        )
    df = pd.read_csv(path)
    cols = md.creditcard_feature_cols(df, config.CREDITCARD_TARGET)
    return md.split_features_target(df, config.CREDITCARD_TARGET, cols)


def candidates(dataset: str, X: pd.DataFrame, y_train: pd.Series) -> dict:
    """Build the three candidate (pipeline, grid, label) triples for a dataset."""
    cols = list(X.columns)

    def wrap(estimator, sampler: str | None = "smote"):
        """Attach the dataset's feature schema to an estimator."""
        if dataset == "fraud":
            return md.fraud_pipeline(estimator, sampler=sampler)
        return md.creditcard_pipeline(estimator, cols, sampler=sampler)

    spw = md.positive_class_weight(y_train)
    return {
        "logreg_smote": (
            wrap(md.build_logistic_regression()),
            md.LOGREG_PARAM_GRID,
            "Logistic Regression + SMOTE",
        ),
        "xgb_smote": (
            wrap(md.build_xgboost()),
            md.XGB_PARAM_GRID,
            "XGBoost + SMOTE",
        ),
        "xgb_weighted": (
            wrap(md.build_xgboost(scale_pos_weight=spw), sampler=None),
            md.XGB_PARAM_GRID,
            f"XGBoost + scale_pos_weight ({spw:.0f})",
        ),
    }


# --------------------------------------------------------------------------- #
# Training protocol
# --------------------------------------------------------------------------- #
def run_dataset(dataset: str, quick: bool = False) -> dict:
    loader = {"fraud": load_fraud, "creditcard": load_creditcard}[dataset]
    X, y = loader()
    if quick:
        X = X.sample(frac=0.1, random_state=config.RANDOM_STATE)
        y = y.loc[X.index]

    log(f"{dataset}: {X.shape[0]:,} rows x {X.shape[1]} features, "
        f"fraud rate {y.mean():.4%}")

    X_train, X_test, y_train, y_test = md.stratified_split(X, y)
    log(f"{dataset}: train {X_train.shape[0]:,} (fraud {y_train.mean():.4%}) | "
        f"test {X_test.shape[0]:,} (fraud {y_test.mean():.4%})")

    test_rows, cv_rows, tuning, curves, mats = [], [], {}, {}, {}

    for key, (pipeline, grid, pretty) in candidates(dataset, X, y_train).items():
        t0 = time.time()
        log(f"{dataset}/{key}: grid-searching {grid} ({TUNE_FOLDS}-fold, AUC-PR)")
        search = md.tune(pipeline, grid, X_train, y_train, cv=TUNE_FOLDS)
        best = search.best_estimator_
        log(f"{dataset}/{key}: best params {search.best_params_} "
            f"(CV AUC-PR {search.best_score_:.4f}) in {time.time() - t0:.0f}s")
        tuning[key] = md.search_results_frame(search).assign(model=key)

        # 5-fold CV of the tuned configuration: the reliability estimate.
        t0 = time.time()
        cv_frame = ev.cross_validate_model(best, X_train, y_train, cv=CV_FOLDS)
        log(f"{dataset}/{key}: {CV_FOLDS}-fold CV done in {time.time() - t0:.0f}s\n"
            f"{cv_frame.round(4).to_string()}")
        cv_rows.append(ev.cv_summary_row(cv_frame, pretty))

        # Operating point chosen on TRAIN out-of-fold scores only.
        thr, thr_f1 = ev.select_threshold(best, X_train, y_train, cv=THRESHOLD_FOLDS)
        log(f"{dataset}/{key}: tuned threshold {thr:.4f} (out-of-fold F1 {thr_f1:.4f})")

        # Single honest evaluation on the held-out, still-imbalanced test set.
        scores = ev.predict_proba_positive(best, X_test)
        test_rows.append(
            ev.classification_metrics(y_test, scores, 0.5, label=f"{pretty} @0.50")
        )
        tuned_row = ev.classification_metrics(
            y_test, scores, thr, label=f"{pretty} @tuned"
        )
        test_rows.append(tuned_row)
        log(f"{dataset}/{key}: TEST auc_pr={tuned_row['auc_pr']:.4f} "
            f"f1={tuned_row['f1']:.4f} precision={tuned_row['precision']:.4f} "
            f"recall={tuned_row['recall']:.4f}")

        curves[pretty] = (y_test.to_numpy(), scores)
        mats[f"{pretty}\n(threshold {thr:.2f})"] = ev.confusion_frame(
            y_test, scores, thr
        )

        config.ensure_dirs()
        out = config.MODELS_DIR / f"{dataset}_{key}.joblib"
        joblib.dump({"pipeline": best, "threshold": thr, "features": list(X.columns)}, out)
        log(f"{dataset}/{key}: saved {out.name}")

    # ---- persist tables & figures ---------------------------------------- #
    test_table = ev.comparison_table(test_rows)
    cv_table = pd.DataFrame(cv_rows)
    tuning_table = pd.concat(tuning.values(), ignore_index=True)

    test_table.to_csv(config.REPORTS_DIR / f"task2_test_metrics_{dataset}.csv", index=False)
    cv_table.to_csv(config.REPORTS_DIR / f"task2_cv_metrics_{dataset}.csv", index=False)
    tuning_table.to_csv(config.REPORTS_DIR / f"task2_tuning_{dataset}.csv", index=False)

    ev.plot_pr_curves(
        curves,
        f"Precision-Recall — {dataset} (held-out test set)",
        config.FIGURES_DIR / f"{dataset}_pr_curves.png",
    )
    ev.plot_confusion_matrices(
        mats,
        f"Confusion matrices — {dataset} (test set, tuned thresholds)",
        config.FIGURES_DIR / f"{dataset}_confusion_matrices.png",
    )

    log(f"{dataset}: TEST METRICS\n{test_table.round(4).to_string(index=False)}")
    log(f"{dataset}: {CV_FOLDS}-FOLD CV (train)\n{cv_table.to_string(index=False)}")

    return {
        "dataset": dataset,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "fraud_rate": float(y.mean()),
        "test": test_table.to_dict(orient="records"),
        "cv": cv_table.to_dict(orient="records"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        choices=["fraud", "creditcard", "both"],
        default="both",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="10%% sample — for verifying the pipeline runs, not for reporting",
    )
    args = ap.parse_args()

    config.ensure_dirs()
    names = ["fraud", "creditcard"] if args.dataset == "both" else [args.dataset]

    summary = {}
    for name in names:
        t0 = time.time()
        summary[name] = run_dataset(name, quick=args.quick)
        log(f"{name}: finished in {(time.time() - t0) / 60:.1f} min")

    suffix = "_quick" if args.quick else ""
    path = config.REPORTS_DIR / f"task2_summary{suffix}.json"
    path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
