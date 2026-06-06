"""Generate the project's Jupyter notebooks programmatically with nbformat.

Keeping notebook *source* in a Python builder makes the notebooks easy to
review in diffs and regenerate deterministically. Run:

    python scripts/build_notebooks.py

then execute them with nbconvert (see notebooks/README.md).
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

# Boilerplate prepended to every notebook so `from src import ...` works
# regardless of the working directory the kernel starts in.
SETUP = """\
import sys
from pathlib import Path

# Make the project root importable so `from src import ...` resolves.
ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)

from src import config
config.ensure_dirs()
FIG = config.FIGURES_DIR
"""


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build(name: str, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    out = NB_DIR / name
    nbf.write(nb, out)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# 1. EDA — Fraud_Data
# --------------------------------------------------------------------------- #
def eda_fraud() -> None:
    cells = [
        md(
            "# EDA — Fraud_Data (E-commerce Transactions)\n\n"
            "**Objective.** Understand the e-commerce transaction data, assess "
            "data quality, characterise the fraud signal, and quantify the class "
            "imbalance that will drive our resampling and evaluation choices.\n\n"
            "All reusable logic lives in `src/` and is unit-tested; this notebook "
            "is the narrative layer on top."
        ),
        code(SETUP),
        code(
            "from src import data_loader as dl, cleaning\n"
            "raw = dl.load_fraud_data()\n"
            "print('Raw shape:', raw.shape)\n"
            "raw.head()"
        ),
        md("## 1. Data overview & types"),
        code("raw.info()"),
        code(
            "# signup_time / purchase_time arrive as strings; ip_address as float.\n"
            "raw.dtypes"
        ),
        md(
            "## 2. Data cleaning\n\n"
            "Steps (each implemented and tested in `src/cleaning.py`):\n"
            "1. **Correct dtypes** — parse timestamps to `datetime`, cast "
            "`source/browser/sex` to `category`.\n"
            "2. **Duplicates** — drop exact duplicate rows.\n"
            "3. **Missing values** — documented policy: drop columns >50% missing, "
            "median-impute numeric, mode-impute categorical."
        ),
        code(
            "print('Missing values per column:')\n"
            "display(cleaning.missing_value_report(raw) if not "
            "cleaning.missing_value_report(raw).empty else 'No missing values.')\n"
            "print('Exact duplicate rows:', raw.duplicated().sum())"
        ),
        code(
            "df = cleaning.clean_fraud_data(raw)\n"
            "print('Cleaned shape:', df.shape)\n"
            "print('Shared devices (device_id used by >1 row):',\n"
            "      (df.groupby('device_id').size() > 1).sum())\n"
            "df.dtypes"
        ),
        md(
            "**Note on duplicates / shared devices.** There are no exact duplicate "
            "rows, but thousands of `device_id`s are shared across transactions — "
            "we deliberately keep these because device-sharing is itself a fraud "
            "signal (engineered later as `device_user_count`)."
        ),
        md(
            "## 3. Class imbalance\n\n"
            "The headline constraint of the project: fraud is a small minority."
        ),
        code(
            "from src.resampling import class_distribution\n"
            "dist = class_distribution(df['class'])\n"
            "display(dist)\n"
            "fraud_rate = df['class'].mean()\n"
            "print(f'Fraud rate: {fraud_rate:.2%}')\n\n"
            "ax = sns.countplot(x='class', data=df)\n"
            "ax.set(title=f'Class balance (fraud = {fraud_rate:.1%})',\n"
            "       xlabel='class (0=legit, 1=fraud)', ylabel='count')\n"
            "plt.savefig(FIG / 'fraud_class_balance.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md(
            "## 4. Univariate distributions\n\n"
            "Key numeric and categorical variables."
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "sns.histplot(df['purchase_value'], bins=40, ax=axes[0])\n"
            "axes[0].set_title('Purchase value ($)')\n"
            "sns.histplot(df['age'], bins=40, ax=axes[1])\n"
            "axes[1].set_title('Age')\n"
            "plt.savefig(FIG / 'fraud_univariate_numeric.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()\n"
            "df[['purchase_value', 'age']].describe()"
        ),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "for ax, col in zip(axes, ['source', 'browser', 'sex']):\n"
            "    order = df[col].value_counts().index\n"
            "    sns.countplot(x=col, data=df, order=order, ax=ax)\n"
            "    ax.set_title(col)\n"
            "    ax.tick_params(axis='x', rotation=30)\n"
            "plt.savefig(FIG / 'fraud_univariate_categorical.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md(
            "## 5. Bivariate relationships with the target\n\n"
            "How does each feature differ between fraudulent and legitimate "
            "transactions?"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "sns.boxplot(x='class', y='purchase_value', data=df, ax=axes[0])\n"
            "axes[0].set_title('Purchase value by class')\n"
            "sns.boxplot(x='class', y='age', data=df, ax=axes[1])\n"
            "axes[1].set_title('Age by class')\n"
            "plt.savefig(FIG / 'fraud_bivariate_numeric.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()"
        ),
        code(
            "# Fraud rate by categorical level — the actionable view.\n"
            "for col in ['source', 'browser', 'sex']:\n"
            "    rate = df.groupby(col, observed=True)['class'].mean().sort_values(ascending=False)\n"
            "    print(f'\\nFraud rate by {col}:')\n"
            "    print((rate * 100).round(2).astype(str) + '%')"
        ),
        md(
            "## 6. Key findings\n\n"
            "- **Imbalance:** ~9.4% of transactions are fraudulent — a minority "
            "class large enough for SMOTE but small enough that accuracy is a "
            "useless metric (a trivial all-legit classifier scores ~90.6%). We "
            "will evaluate with **AUC-PR, recall, precision and F1**.\n"
            "- **Purchase value / age** distributions overlap substantially "
            "between classes — no single numeric feature cleanly separates fraud, "
            "motivating the engineered time/velocity features in "
            "`feature-engineering.ipynb`.\n"
            "- **Channel/browser** fraud rates vary, providing categorical signal "
            "(one-hot encoded downstream).\n"
            "- **Device sharing** is widespread and retained as a fraud signal."
        ),
    ]
    build("eda-fraud-data.ipynb", cells)


# --------------------------------------------------------------------------- #
# 2. EDA — creditcard
# --------------------------------------------------------------------------- #
def eda_creditcard() -> None:
    cells = [
        md(
            "# EDA — Credit-Card Transactions\n\n"
            "**Objective.** Characterise the bank credit-card dataset (PCA-"
            "anonymised features `V1–V28` plus `Time` and `Amount`) and quantify "
            "its extreme class imbalance.\n\n"
            "> This notebook requires `data/raw/creditcard.csv`. If the file is "
            "absent the notebook short-circuits with a clear message so the rest "
            "of the pipeline still executes."
        ),
        code(SETUP),
        code(
            "from src import data_loader as dl, cleaning\n"
            "from src.resampling import class_distribution\n\n"
            "HAVE_CC = config.CREDITCARD_RAW.exists()\n"
            "if not HAVE_CC:\n"
            "    print('creditcard.csv not found in data/raw/ — skipping analysis.')\n"
            "else:\n"
            "    cc = dl.load_creditcard()\n"
            "    print('Raw shape:', cc.shape)\n"
            "    display(cc.head())"
        ),
        md("## 1. Overview, dtypes & data quality"),
        code(
            "if HAVE_CC:\n"
            "    cc.info()\n"
            "    print('\\nMissing values:')\n"
            "    print(cleaning.missing_value_report(cc) if not "
            "cleaning.missing_value_report(cc).empty else 'None')\n"
            "    print('Exact duplicate rows:', cc.duplicated().sum())"
        ),
        code(
            "if HAVE_CC:\n"
            "    cc = cleaning.clean_creditcard(cc)\n"
            "    print('Cleaned shape (post dedup):', cc.shape)"
        ),
        md("## 2. Class imbalance"),
        code(
            "if HAVE_CC:\n"
            "    dist = class_distribution(cc['Class'])\n"
            "    display(dist)\n"
            "    rate = cc['Class'].mean()\n"
            "    print(f'Fraud rate: {rate:.4%}')\n"
            "    ax = sns.countplot(x='Class', data=cc)\n"
            "    ax.set(title=f'Credit-card class balance (fraud={rate:.3%})', yscale='log')\n"
            "    plt.savefig(FIG / 'cc_class_balance.png', dpi=120, bbox_inches='tight')\n"
            "    plt.show()"
        ),
        md("## 3. Univariate: Amount and Time"),
        code(
            "if HAVE_CC:\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "    sns.histplot(cc['Amount'], bins=50, ax=axes[0])\n"
            "    axes[0].set(title='Transaction amount', yscale='log')\n"
            "    sns.histplot(cc['Time'] / 3600, bins=48, ax=axes[1])\n"
            "    axes[1].set(title='Time (hours since first txn)')\n"
            "    plt.savefig(FIG / 'cc_univariate.png', dpi=120, bbox_inches='tight')\n"
            "    plt.show()\n"
            "    display(cc[['Amount', 'Time']].describe())"
        ),
        md("## 4. Bivariate: Amount vs class & feature correlations with target"),
        code(
            "if HAVE_CC:\n"
            "    fig, ax = plt.subplots(figsize=(6, 4))\n"
            "    sns.boxplot(x='Class', y='Amount', data=cc, ax=ax)\n"
            "    ax.set(title='Amount by class', yscale='log')\n"
            "    plt.savefig(FIG / 'cc_amount_by_class.png', dpi=120, bbox_inches='tight')\n"
            "    plt.show()"
        ),
        code(
            "if HAVE_CC:\n"
            "    corr = cc.corr(numeric_only=True)['Class'].drop('Class').sort_values()\n"
            "    fig, ax = plt.subplots(figsize=(6, 8))\n"
            "    corr.plot.barh(ax=ax)\n"
            "    ax.set_title('Linear correlation of each feature with Class')\n"
            "    plt.savefig(FIG / 'cc_target_correlation.png', dpi=120, bbox_inches='tight')\n"
            "    plt.show()\n"
            "    print('Most negatively/positively correlated PCA features:')\n"
            "    print(pd.concat([corr.head(4), corr.tail(4)]))"
        ),
        md(
            "## 5. Key findings\n\n"
            "- **Extreme imbalance** (~0.17% fraud) — far more severe than "
            "Fraud_Data. Undersampling would discard ~99% of data; **SMOTE / "
            "SMOTEENN on the training fold only** is the appropriate response, and "
            "**AUC-PR** is the primary metric.\n"
            "- `Amount` is heavily right-skewed and should be **scaled** (it is the "
            "only non-PCA numeric besides `Time`).\n"
            "- Several PCA components (e.g. V14, V12, V17, V10) carry strong linear "
            "signal toward the target."
        ),
    ]
    build("eda-creditcard.ipynb", cells)


# --------------------------------------------------------------------------- #
# 3. Feature engineering (Fraud_Data) — incl. geolocation, scaling, encoding, SMOTE
# --------------------------------------------------------------------------- #
def feature_engineering_nb() -> None:
    cells = [
        md(
            "# Feature Engineering — Fraud_Data\n\n"
            "**Objective.** Turn the cleaned transactions into a model-ready "
            "matrix: geolocation, time & velocity features, scaling, one-hot "
            "encoding, a leakage-safe train/test split, and SMOTE resampling on "
            "the training set only.\n\n"
            "Output: `data/processed/fraud_features.csv`."
        ),
        code(SETUP),
        code(
            "from src import data_loader as dl, cleaning, feature_engineering as fe, "
            "geolocation as geo\n"
            "df = cleaning.clean_fraud_data(dl.load_fraud_data())\n"
            "print('Cleaned:', df.shape)"
        ),
        md(
            "## 1. Geolocation integration\n\n"
            "Convert IPs to integers and map each to a country via **range-based "
            "lookup** (`np.searchsorted`, O(n log m)). Requires "
            "`data/raw/IpAddress_to_Country.csv`."
        ),
        code(
            "if config.IP_COUNTRY_RAW.exists():\n"
            "    ip_country = dl.load_ip_country()\n"
            "    df = geo.add_geolocation(df, ip_country)\n"
            "    matched = (df['country'] != geo.UNKNOWN_COUNTRY).mean()\n"
            "    print(f'IP->country match rate: {matched:.1%}')\n"
            "    display(df[['ip_address', 'ip_int', 'country']].head())\n"
            "else:\n"
            "    print('IpAddress_to_Country.csv not found — adding ip_int only.')\n"
            "    df = geo.add_ip_integer(df)\n"
            "    df['country'] = geo.UNKNOWN_COUNTRY"
        ),
        code(
            "# Fraud patterns by country (top by volume).\n"
            "if (df['country'] != geo.UNKNOWN_COUNTRY).any():\n"
            "    rates = geo.fraud_rate_by_country(df, min_count=100)\n"
            "    print('Highest-fraud-rate countries (>=100 txns):')\n"
            "    display(rates.head(10))\n"
            "    top = rates.head(15)\n"
            "    fig, ax = plt.subplots(figsize=(8, 5))\n"
            "    sns.barplot(y=top.index, x=top['fraud_rate'], ax=ax)\n"
            "    ax.set(title='Fraud rate by country (top 15, >=100 txns)', xlabel='fraud rate')\n"
            "    plt.savefig(FIG / 'fraud_rate_by_country.png', dpi=120, bbox_inches='tight')\n"
            "    plt.show()"
        ),
        md(
            "## 2. Time features\n\n"
            "`hour_of_day`, `day_of_week`, and `time_since_signup` (hours between "
            "signup and purchase — near-zero gaps flag automated fraud)."
        ),
        code(
            "df = fe.add_time_features(df)\n"
            "display(df[['signup_time','purchase_time','hour_of_day','day_of_week','time_since_signup']].head())\n"
            "# time_since_signup differs sharply by class:\n"
            "print(df.groupby('class')['time_since_signup'].median().rename('median_hours'))"
        ),
        md(
            "## 3. Transaction frequency & velocity\n\n"
            "Per-user and per-device counts, distinct users per device, and a "
            "24-hour rolling purchase velocity per device."
        ),
        code(
            "df = fe.add_frequency_features(df)\n"
            "df = fe.add_velocity_features(df, window_hours=24.0)\n"
            "vel_cols = ['user_transaction_count','device_transaction_count',\n"
            "            'device_user_count','device_velocity_24h']\n"
            "display(df[vel_cols].describe().round(2))\n"
            "print('\\nMean of each velocity feature by class:')\n"
            "display(df.groupby('class')[vel_cols].mean().round(2))"
        ),
        md(
            "## 4. Train/test split (before scaling & resampling)\n\n"
            "We split **first** so that scaler fitting and SMOTE see only training "
            "data — preventing leakage. Stratified to preserve the fraud rate."
        ),
        code(
            "from sklearn.model_selection import train_test_split\n"
            "from src import transform, resampling\n\n"
            "numeric_cols = ['purchase_value','age','hour_of_day','day_of_week',\n"
            "                'time_since_signup','user_transaction_count',\n"
            "                'device_transaction_count','device_user_count',\n"
            "                'device_velocity_24h']\n"
            "categorical_cols = ['source','browser','sex']\n"
            "feature_cols = numeric_cols + categorical_cols\n\n"
            "X = df[feature_cols].copy()\n"
            "for c in categorical_cols:\n"
            "    X[c] = X[c].astype(str)\n"
            "y = df['class']\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)\n"
            "print('Train:', X_train.shape, 'Test:', X_test.shape)"
        ),
        md(
            "## 5. Scaling + one-hot encoding\n\n"
            "`StandardScaler` for numerics, `OneHotEncoder` for categoricals, "
            "bundled in a `ColumnTransformer` **fit on the training set only**."
        ),
        code(
            "pre = transform.build_preprocessor(numeric_cols, categorical_cols, scaler='standard')\n"
            "X_train_t = transform.fit_transform_frame(pre, X_train)\n"
            "X_test_t = transform.transform_frame(pre, X_test)\n"
            "print('Transformed train matrix:', X_train_t.shape)\n"
            "X_train_t.head()"
        ),
        md(
            "## 6. Handle class imbalance — SMOTE (training set only)\n\n"
            "**Why SMOTE over undersampling?** With ~9% fraud, undersampling the "
            "majority would discard the bulk of legitimate transactions and the "
            "signal they carry. Random oversampling merely duplicates minority "
            "rows (overfitting risk). SMOTE synthesises new minority points by "
            "interpolating between near neighbours, enriching the minority region "
            "without literal duplicates. **Applied to the training fold only** so "
            "the test set keeps the real-world distribution."
        ),
        code(
            "X_train_res, y_train_res = resampling.resample(X_train_t, y_train, method='smote')\n"
            "report = resampling.resample_report(y_train, y_train_res)\n"
            "print('Class distribution before vs after SMOTE (TRAIN ONLY):')\n"
            "display(report)\n"
            "print('Test set distribution is left UNCHANGED:')\n"
            "display(resampling.class_distribution(y_test))"
        ),
        md("## 7. Persist processed features"),
        code(
            "# Save the engineered (pre-scaling) frame for reuse + the split sizes.\n"
            "out_cols = feature_cols + ['country', 'class']\n"
            "df[out_cols].to_csv(config.FRAUD_FEATURES, index=False)\n"
            "print('Saved', config.FRAUD_FEATURES)\n"
            "print('Engineered feature columns:', feature_cols)"
        ),
        md(
            "## 8. Summary\n\n"
            "- **Geolocation:** IPs mapped to countries via range lookup; fraud "
            "rate varies by country.\n"
            "- **Engineered features:** `hour_of_day`, `day_of_week`, "
            "`time_since_signup`, per-user/device counts, `device_user_count`, "
            "`device_velocity_24h`.\n"
            "- **Transformation:** standardised numerics + one-hot categoricals via "
            "a leakage-safe `ColumnTransformer`.\n"
            "- **Imbalance:** SMOTE balanced the **training** set "
            "(~90/10 → 50/50); the **test** set was untouched for honest "
            "evaluation."
        ),
    ]
    build("feature-engineering.ipynb", cells)


# --------------------------------------------------------------------------- #
# 4 & 5. Modeling + SHAP placeholders (scoped for later tasks)
# --------------------------------------------------------------------------- #
def modeling_nb() -> None:
    cells = [
        md(
            "# Modeling — Fraud Detection (placeholder for Task 2)\n\n"
            "Task 1 delivers clean, feature-engineered, resampled data. This "
            "notebook is scaffolded for the modeling task: train a baseline "
            "(Logistic Regression) and an ensemble (Random Forest / Gradient "
            "Boosting), evaluate with **AUC-PR, F1, recall, precision** (not "
            "accuracy), and compare.\n\n"
            "The data preparation below reuses the exact `src/` pipeline from "
            "`feature-engineering.ipynb`."
        ),
        code(SETUP),
        code(
            "# Reuse the tested pipeline. Uncomment to build the modeling matrix.\n"
            "# from src import data_loader as dl, cleaning, feature_engineering as fe, transform, resampling\n"
            "# ... (see feature-engineering.ipynb for the full prep)\n"
            "print('Scaffold ready — implement models in Task 2.')"
        ),
    ]
    build("modeling.ipynb", cells)


def shap_nb() -> None:
    cells = [
        md(
            "# Model Explainability with SHAP (placeholder for Task 3)\n\n"
            "Once a model is trained (Task 2), this notebook will use SHAP to "
            "explain global feature importance (summary plot) and individual "
            "predictions (force/waterfall plots) — e.g. confirming whether "
            "`time_since_signup` and `device_velocity_24h` drive fraud scores."
        ),
        code(SETUP),
        code(
            "import shap\n"
            "print('shap', shap.__version__, '— scaffold ready for Task 3.')"
        ),
    ]
    build("shap-explainability.ipynb", cells)


if __name__ == "__main__":
    eda_fraud()
    eda_creditcard()
    feature_engineering_nb()
    modeling_nb()
    shap_nb()
    print("All notebooks generated.")
