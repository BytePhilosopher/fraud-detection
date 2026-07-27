# Notebooks

Narrative analysis layer. All reusable logic lives in [`src/`](../src) (unit-tested);
notebooks call into it so they stay readable and reproducible.

| Notebook | Purpose | Task |
|---|---|---|
| `eda-fraud-data.ipynb` | Clean Fraud_Data, univariate/bivariate EDA, class-imbalance quantification | 1 |
| `eda-creditcard.ipynb` | EDA + extreme-imbalance analysis for the credit-card data | 1 |
| `feature-engineering.ipynb` | Geolocation merge, time/velocity features, scaling, encoding, train/test split, SMOTE | 1 |
| `modeling.ipynb` | Stratified split, Logistic Regression baseline, XGBoost ensemble, 5-fold CV, threshold selection, model comparison | 2 |
| `shap-explainability.ipynb` | Built-in importance, SHAP summary/force/waterfall, effect shapes, scored rules | 3 |

`modeling.ipynb` narrates the protocol using the winning hyperparameters recorded
in `reports/task2_tuning_*.csv`. The exhaustive grid searches that produce those
files live in [`scripts/train_models.py`](../scripts/train_models.py):

```bash
python scripts/train_models.py            # both datasets (~26 min)
python scripts/train_models.py --quick    # 10% sample, verifies the pipeline
```

`shap-explainability.ipynb` loads the models saved by that run
(`models/{dataset}_selected.joblib`) and rebuilds the same test split, so its
explanations describe the exact predictions Task 2 reported.
[`scripts/explain_models.py`](../scripts/explain_models.py) writes every Task-3
figure and table in one batch (~4 min).

## Regenerate & execute

The notebooks are generated deterministically from [`scripts/build_notebooks.py`](../scripts/build_notebooks.py):

```bash
python scripts/build_notebooks.py          # (re)write the .ipynb files
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/*.ipynb
```

Requires the three raw files in [`data/raw/`](../data/raw). Figures are written to
`reports/figures/`; the processed feature table to `data/processed/`.
