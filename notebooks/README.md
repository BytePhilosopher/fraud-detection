# Notebooks

Narrative analysis layer. All reusable logic lives in [`src/`](../src) (unit-tested);
notebooks call into it so they stay readable and reproducible.

| Notebook | Purpose | Task |
|---|---|---|
| `eda-fraud-data.ipynb` | Clean Fraud_Data, univariate/bivariate EDA, class-imbalance quantification | 1 |
| `eda-creditcard.ipynb` | EDA + extreme-imbalance analysis for the credit-card data | 1 |
| `feature-engineering.ipynb` | Geolocation merge, time/velocity features, scaling, encoding, train/test split, SMOTE | 1 |
| `modeling.ipynb` | Baseline + ensemble models, AUC-PR/F1/recall evaluation (scaffold) | 2 |
| `shap-explainability.ipynb` | Global & local SHAP explanations (scaffold) | 3 |

## Regenerate & execute

The notebooks are generated deterministically from [`scripts/build_notebooks.py`](../scripts/build_notebooks.py):

```bash
python scripts/build_notebooks.py          # (re)write the .ipynb files
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/*.ipynb
```

Requires the three raw files in [`data/raw/`](../data/raw). Figures are written to
`reports/figures/`; the processed feature table to `data/processed/`.
