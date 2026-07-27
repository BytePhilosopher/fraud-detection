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
# 4. Modeling (Task 2)
# --------------------------------------------------------------------------- #
def modeling_nb() -> None:
    cells = [
        md(
            "# Modeling — Task 2\n\n"
            "**Objective.** Build, train and compare classification models for "
            "fraud detection on two severely imbalanced datasets, then select one "
            "with a justified trade-off between performance and interpretability."
            "\n\n"
            "**Protocol** (identical for both datasets, implemented in "
            "[`src/modeling.py`](../src/modeling.py) and "
            "[`src/evaluation.py`](../src/evaluation.py)):\n\n"
            "1. Separate features from target, then a **stratified 80/20 split** "
            "so both halves keep the real fraud rate.\n"
            "2. Wrap scaling + encoding + **SMOTE** + estimator in a single "
            "`imblearn` pipeline. SMOTE runs on `fit` only — never on `predict` — "
            "so every CV fold resamples its own training portion and the test set "
            "stays at production prevalence.\n"
            "3. **Baseline:** Logistic Regression. **Ensemble:** XGBoost, with a "
            "small grid search on `n_estimators` / `max_depth` / `learning_rate` "
            "scored by AUC-PR.\n"
            "4. **5-fold stratified CV** on the training half for mean ± std.\n"
            "5. Pick the decision threshold from **out-of-fold training** "
            "probabilities, then score once on the held-out test set.\n\n"
            "**Metrics.** AUC-PR (primary), F1, precision, recall, confusion "
            "matrix. Accuracy is excluded: predicting \"legitimate\" every time "
            "scores 90.6% / 99.83% and catches no fraud.\n\n"
            "> The exhaustive grid searches live in "
            "[`scripts/train_models.py`](../scripts/train_models.py) (~1 h). This "
            "notebook narrates the protocol using the winning hyperparameters that "
            "run recorded in `reports/task2_tuning_*.csv`, so it executes in "
            "minutes and reproduces the same numbers."
        ),
        code(SETUP),
        code(
            "import joblib\n"
            "from src import modeling as md, evaluation as ev\n\n"
            "CV_FOLDS = 5\n\n"
            "def best_params(dataset, model_key, fallback):\n"
            "    \"\"\"Winning hyperparameters from scripts/train_models.py.\n\n"
            "    Read from the recorded tuning table when present so the notebook\n"
            "    tracks the real search; `fallback` keeps it runnable standalone.\n"
            "    \"\"\"\n"
            "    path = config.REPORTS_DIR / f'task2_tuning_{dataset}.csv'\n"
            "    if not path.exists():\n"
            "        return fallback\n"
            "    tbl = pd.read_csv(path)\n"
            "    top = tbl[(tbl['model'] == model_key) & (tbl['rank_test_score'] == 1)]\n"
            "    if top.empty:\n"
            "        return fallback\n"
            "    row = top.iloc[0]\n"
            "    out = {}\n"
            "    for col in [c for c in tbl.columns if c.startswith('param_clf__')]:\n"
            "        val = row[col]\n"
            "        if pd.notna(val):\n"
            "            out[col.replace('param_clf__', '')] = val\n"
            "    return out or fallback\n\n"
            "print('helpers ready')"
        ),
        md(
            "## 1. Fraud_Data — features and target\n\n"
            "Features are the Task-1 engineered columns plus the geolocated "
            "`country`. `country` has 182 levels, so the encoder folds levels "
            "below 1% frequency into one *infrequent* column instead of emitting "
            "a long tail of near-empty dummies."
        ),
        code(
            "fraud = pd.read_csv(config.FRAUD_FEATURES)\n"
            "feature_cols = md.FRAUD_NUMERIC_COLS + md.FRAUD_CATEGORICAL_COLS\n"
            "X, y = md.split_features_target(fraud, config.FRAUD_TARGET, feature_cols)\n"
            "for c in md.FRAUD_CATEGORICAL_COLS:\n"
            "    X[c] = X[c].astype(str)\n"
            "print('X:', X.shape, '| target:', config.FRAUD_TARGET)\n"
            "print(f'fraud rate: {y.mean():.4%}')\n"
            "X.head()"
        ),
        md(
            "## 2. Stratified train/test split\n\n"
            "Stratifying matters most on the credit-card set (473 frauds total) but "
            "we apply it to both so the reported metrics are comparable to "
            "production prevalence."
        ),
        code(
            "X_train, X_test, y_train, y_test = md.stratified_split(X, y, test_size=0.2)\n"
            "print(f'train {X_train.shape[0]:,} rows, fraud {y_train.mean():.4%}')\n"
            "print(f'test  {X_test.shape[0]:,} rows, fraud {y_test.mean():.4%}')\n"
            "print('\\nSplit preserved the class ratio — no resampling has happened yet.')"
        ),
        md(
            "## 3. Baseline — Logistic Regression + SMOTE\n\n"
            "A linear model is the right baseline here: coefficients are directly "
            "readable (each is a log-odds contribution), it trains in seconds, and "
            "it sets the bar any ensemble must clear to justify its opacity."
        ),
        code(
            "lr_params = best_params('fraud', 'logreg_smote', {'C': 1.0})\n"
            "lr = md.fraud_pipeline(md.build_logistic_regression(C=float(lr_params['C'])))\n"
            "lr.fit(X_train, y_train)\n"
            "lr_scores = ev.predict_proba_positive(lr, X_test)\n"
            "print('tuned C =', lr_params['C'])\n"
            "pd.Series(ev.classification_metrics(y_test, lr_scores)).to_frame('logreg @0.50').round(4)"
        ),
        md(
            "### Why the default 0.50 threshold is wrong after SMOTE\n\n"
            "SMOTE balances the training set, so the model's scores are calibrated "
            "to a 50/50 world, not a 9% one — 0.50 therefore over-predicts fraud. "
            "We choose the threshold from **out-of-fold training** probabilities "
            "(`ev.select_threshold`), which never touches the test set."
        ),
        code(
            "lr_thr, lr_oof_f1 = ev.select_threshold(lr, X_train, y_train, cv=3)\n"
            "print(f'threshold from out-of-fold train scores: {lr_thr:.4f} "
            "(out-of-fold F1 {lr_oof_f1:.4f})')\n"
            "lr_metrics = ev.classification_metrics(y_test, lr_scores, lr_thr)\n"
            "display(pd.Series(lr_metrics).to_frame('logreg @tuned').round(4))\n"
            "print('\\nConfusion matrix (test set):')\n"
            "display(ev.confusion_frame(y_test, lr_scores, lr_thr))"
        ),
        md(
            "### Baseline interpretability\n\n"
            "The coefficients are the reason to keep a linear model in the running: "
            "each one is an auditable statement about the direction and size of a "
            "feature's effect."
        ),
        code(
            "names = lr.named_steps['preprocess'].get_feature_names_out()\n"
            "coefs = pd.Series(lr.named_steps['clf'].coef_[0], index=names)\n"
            "top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(12)\n"
            "display(top.to_frame('coefficient (log-odds)').round(3))\n"
            "fig, ax = plt.subplots(figsize=(7, 5))\n"
            "top.sort_values().plot.barh(ax=ax, color=np.where(top.sort_values() > 0, 'firebrick', 'steelblue'))\n"
            "ax.set(title='Logistic Regression — largest coefficients', xlabel='log-odds')\n"
            "plt.savefig(FIG / 'fraud_logreg_coefficients.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md(
            "## 4. Ensemble — XGBoost\n\n"
            "Gradient-boosted trees capture the interactions a linear model cannot "
            "— e.g. *low `time_since_signup` **and** high `device_user_count`* is "
            "far more suspicious than either alone.\n\n"
            "Two imbalance strategies are compared on identical splits:\n\n"
            "| Variant | Imbalance handling |\n"
            "|---|---|\n"
            "| `xgb_smote` | SMOTE synthesises minority rows inside each fold |\n"
            "| `xgb_weighted` | `scale_pos_weight = n_neg/n_pos` reweights the gradient |\n\n"
            "Weighting is cheaper (no synthetic rows) and, for trees, usually at "
            "least as good — worth measuring rather than assuming."
        ),
        code(
            "xgb_grid_fallback = {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.05}\n"
            "xgb_p = best_params('fraud', 'xgb_smote', xgb_grid_fallback)\n"
            "xgb_p = {k: (int(v) if k != 'learning_rate' else float(v)) for k, v in xgb_p.items()}\n"
            "print('tuned XGBoost params:', xgb_p)\n\n"
            "xgb = md.fraud_pipeline(md.build_xgboost(**xgb_p))\n"
            "xgb.fit(X_train, y_train)\n"
            "xgb_scores = ev.predict_proba_positive(xgb, X_test)\n"
            "xgb_thr, _ = ev.select_threshold(xgb, X_train, y_train, cv=3)\n"
            "xgb_metrics = ev.classification_metrics(y_test, xgb_scores, xgb_thr)\n"
            "display(pd.Series(xgb_metrics).to_frame('xgb+smote @tuned').round(4))\n"
            "display(ev.confusion_frame(y_test, xgb_scores, xgb_thr))"
        ),
        code(
            "# Cost-weighted variant: no SMOTE, reweighted positives instead.\n"
            "spw = md.positive_class_weight(y_train)\n"
            "xgb_w_p = best_params('fraud', 'xgb_weighted', xgb_grid_fallback)\n"
            "xgb_w_p = {k: (int(v) if k != 'learning_rate' else float(v)) for k, v in xgb_w_p.items()}\n"
            "xgb_w = md.fraud_pipeline(\n"
            "    md.build_xgboost(scale_pos_weight=spw, **xgb_w_p), sampler=None)\n"
            "xgb_w.fit(X_train, y_train)\n"
            "xgb_w_scores = ev.predict_proba_positive(xgb_w, X_test)\n"
            "xgb_w_thr, _ = ev.select_threshold(xgb_w, X_train, y_train, cv=3)\n"
            "xgb_w_metrics = ev.classification_metrics(y_test, xgb_w_scores, xgb_w_thr)\n"
            "print(f'scale_pos_weight = {spw:.2f}')\n"
            "display(pd.Series(xgb_w_metrics).to_frame('xgb+weight @tuned').round(4))"
        ),
        md(
            "## 5. Cross-validation — 5-fold stratified\n\n"
            "A single split can flatter or punish a model by luck. Each fold refits "
            "the **whole pipeline** (scaler, encoder, SMOTE, model), so the spread "
            "below is an honest stability estimate rather than a leakage artefact."
        ),
        code(
            "cv_rows = []\n"
            "for label, pipe in [('Logistic Regression + SMOTE', lr),\n"
            "                    ('XGBoost + SMOTE', xgb),\n"
            "                    ('XGBoost + scale_pos_weight', xgb_w)]:\n"
            "    frame = ev.cross_validate_model(pipe, X_train, y_train, cv=CV_FOLDS)\n"
            "    print(f'--- {label} ---')\n"
            "    display(frame.round(4))\n"
            "    cv_rows.append(ev.cv_summary_row(frame, label))\n"
            "fraud_cv = pd.DataFrame(cv_rows)\n"
            "print('5-fold CV on the training half (mean ± std):')\n"
            "fraud_cv"
        ),
        md("## 6. Fraud_Data comparison"),
        code(
            "fraud_test = ev.comparison_table([\n"
            "    {**lr_metrics, 'model': 'Logistic Regression + SMOTE'},\n"
            "    {**xgb_metrics, 'model': 'XGBoost + SMOTE'},\n"
            "    {**xgb_w_metrics, 'model': 'XGBoost + scale_pos_weight'},\n"
            "])\n"
            "display(fraud_test.round(4))\n"
            "ev.plot_pr_curves(\n"
            "    {'Logistic Regression + SMOTE': (y_test.to_numpy(), lr_scores),\n"
            "     'XGBoost + SMOTE': (y_test.to_numpy(), xgb_scores),\n"
            "     'XGBoost + scale_pos_weight': (y_test.to_numpy(), xgb_w_scores)},\n"
            "    'Precision-Recall — Fraud_Data (test set)',\n"
            "    FIG / 'fraud_pr_curves_nb.png')\n"
            "plt.show()"
        ),
        md(
            "## 7. creditcard — the same protocol at 0.17% fraud\n\n"
            "The features are already anonymised PCA components, so there is no "
            "categorical encoding: the pipeline is scale → SMOTE → model. With only "
            "**473 frauds in 283,726 rows**, this is where AUC-PR earns its keep — "
            "ROC-AUC looks excellent for models that are barely usable."
        ),
        code(
            "cc = pd.read_csv(config.CREDITCARD_CLEAN)\n"
            "cc_cols = md.creditcard_feature_cols(cc, config.CREDITCARD_TARGET)\n"
            "Xc, yc = md.split_features_target(cc, config.CREDITCARD_TARGET, cc_cols)\n"
            "Xc_train, Xc_test, yc_train, yc_test = md.stratified_split(Xc, yc, test_size=0.2)\n"
            "print(f'{Xc.shape[0]:,} rows x {Xc.shape[1]} features | fraud {yc.mean():.4%}')\n"
            "print(f'train frauds: {int(yc_train.sum())} | test frauds: {int(yc_test.sum())}')"
        ),
        code(
            "cc_lr_p = best_params('creditcard', 'logreg_smote', {'C': 1.0})\n"
            "cc_lr = md.creditcard_pipeline(\n"
            "    md.build_logistic_regression(C=float(cc_lr_p['C'])), cc_cols)\n"
            "cc_lr.fit(Xc_train, yc_train)\n"
            "cc_lr_scores = ev.predict_proba_positive(cc_lr, Xc_test)\n"
            "cc_lr_thr, _ = ev.select_threshold(cc_lr, Xc_train, yc_train, cv=3)\n"
            "cc_lr_metrics = ev.classification_metrics(yc_test, cc_lr_scores, cc_lr_thr)\n"
            "print('Baseline at the default 0.50 vs the tuned threshold:')\n"
            "display(pd.DataFrame([\n"
            "    ev.classification_metrics(yc_test, cc_lr_scores, 0.5, label='logreg @0.50'),\n"
            "    {**cc_lr_metrics, 'model': f'logreg @{cc_lr_thr:.3f}'},\n"
            "]).set_index('model').round(4))\n"
            "print('At 0.50 the baseline flags ~1,500 legitimate transactions to catch ~83 frauds —')\n"
            "print('precision ~5%. Threshold choice is not cosmetic on this dataset.')"
        ),
        code(
            "cc_xgb_p = best_params('creditcard', 'xgb_smote', xgb_grid_fallback)\n"
            "cc_xgb_p = {k: (int(v) if k != 'learning_rate' else float(v)) for k, v in cc_xgb_p.items()}\n"
            "cc_xgb = md.creditcard_pipeline(md.build_xgboost(**cc_xgb_p), cc_cols)\n"
            "cc_xgb.fit(Xc_train, yc_train)\n"
            "cc_xgb_scores = ev.predict_proba_positive(cc_xgb, Xc_test)\n"
            "cc_xgb_thr, _ = ev.select_threshold(cc_xgb, Xc_train, yc_train, cv=3)\n"
            "cc_xgb_metrics = ev.classification_metrics(yc_test, cc_xgb_scores, cc_xgb_thr)\n\n"
            "cc_spw = md.positive_class_weight(yc_train)\n"
            "cc_xgb_w_p = best_params('creditcard', 'xgb_weighted', xgb_grid_fallback)\n"
            "cc_xgb_w_p = {k: (int(v) if k != 'learning_rate' else float(v)) for k, v in cc_xgb_w_p.items()}\n"
            "cc_xgb_w = md.creditcard_pipeline(\n"
            "    md.build_xgboost(scale_pos_weight=cc_spw, **cc_xgb_w_p), cc_cols, sampler=None)\n"
            "cc_xgb_w.fit(Xc_train, yc_train)\n"
            "cc_xgb_w_scores = ev.predict_proba_positive(cc_xgb_w, Xc_test)\n"
            "cc_xgb_w_thr, _ = ev.select_threshold(cc_xgb_w, Xc_train, yc_train, cv=3)\n"
            "cc_xgb_w_metrics = ev.classification_metrics(yc_test, cc_xgb_w_scores, cc_xgb_w_thr)\n"
            "print(f'scale_pos_weight = {cc_spw:.1f}')\n"
            "cc_test = ev.comparison_table([\n"
            "    {**cc_lr_metrics, 'model': 'Logistic Regression + SMOTE'},\n"
            "    {**cc_xgb_metrics, 'model': 'XGBoost + SMOTE'},\n"
            "    {**cc_xgb_w_metrics, 'model': 'XGBoost + scale_pos_weight'},\n"
            "])\n"
            "cc_test.round(4)"
        ),
        code(
            "cc_cv_rows = []\n"
            "for label, pipe in [('Logistic Regression + SMOTE', cc_lr),\n"
            "                    ('XGBoost + SMOTE', cc_xgb),\n"
            "                    ('XGBoost + scale_pos_weight', cc_xgb_w)]:\n"
            "    frame = ev.cross_validate_model(pipe, Xc_train, yc_train, cv=CV_FOLDS)\n"
            "    cc_cv_rows.append(ev.cv_summary_row(frame, label))\n"
            "cc_cv = pd.DataFrame(cc_cv_rows)\n"
            "print('creditcard — 5-fold CV on the training half (mean ± std):')\n"
            "display(cc_cv)\n"
            "ev.plot_pr_curves(\n"
            "    {'Logistic Regression + SMOTE': (yc_test.to_numpy(), cc_lr_scores),\n"
            "     'XGBoost + SMOTE': (yc_test.to_numpy(), cc_xgb_scores),\n"
            "     'XGBoost + scale_pos_weight': (yc_test.to_numpy(), cc_xgb_w_scores)},\n"
            "    'Precision-Recall — creditcard (test set)',\n"
            "    FIG / 'creditcard_pr_curves_nb.png')\n"
            "plt.show()\n"
            "ev.plot_confusion_matrices(\n"
            "    {'Logistic Regression': ev.confusion_frame(yc_test, cc_lr_scores, cc_lr_thr),\n"
            "     'XGBoost + SMOTE': ev.confusion_frame(yc_test, cc_xgb_scores, cc_xgb_thr),\n"
            "     'XGBoost + weight': ev.confusion_frame(yc_test, cc_xgb_w_scores, cc_xgb_w_thr)},\n"
            "    'creditcard — confusion matrices (test set, tuned thresholds)',\n"
            "    FIG / 'creditcard_confusion_nb.png')\n"
            "plt.show()"
        ),
        md(
            "## 8. Model selection\n\n"
            "The two datasets are scored side by side below. Selection rule: take "
            "the highest **AUC-PR**, then break ties on **interpretability** and "
            "**cost of the residual errors** (a false negative is a realised loss; "
            "a false positive is a blocked customer)."
        ),
        code(
            "combined = pd.concat([\n"
            "    fraud_test.assign(dataset='Fraud_Data'),\n"
            "    cc_test.assign(dataset='creditcard'),\n"
            "])[['dataset', 'model', 'auc_pr', 'f1', 'precision', 'recall',\n"
            "    'roc_auc', 'threshold', 'tp', 'fp', 'fn', 'tn']]\n"
            "display(combined.round(4).set_index(['dataset', 'model']))\n"
            "print('\\n5-fold CV, Fraud_Data:'); display(fraud_cv)\n"
            "print('5-fold CV, creditcard:'); display(cc_cv)"
        ),
        code(
            "# Persist the selected model per dataset (pipeline + its threshold).\n"
            "config.ensure_dirs()\n"
            "best_fraud = max([(xgb_metrics['auc_pr'], 'xgb_smote', xgb, xgb_thr),\n"
            "                  (xgb_w_metrics['auc_pr'], 'xgb_weighted', xgb_w, xgb_w_thr),\n"
            "                  (lr_metrics['auc_pr'], 'logreg_smote', lr, lr_thr)])\n"
            "best_cc = max([(cc_xgb_metrics['auc_pr'], 'xgb_smote', cc_xgb, cc_xgb_thr),\n"
            "               (cc_xgb_w_metrics['auc_pr'], 'xgb_weighted', cc_xgb_w, cc_xgb_w_thr),\n"
            "               (cc_lr_metrics['auc_pr'], 'logreg_smote', cc_lr, cc_lr_thr)])\n"
            "for dataset, (score, key, pipe, thr) in [('fraud', best_fraud), ('creditcard', best_cc)]:\n"
            "    out = config.MODELS_DIR / f'{dataset}_selected.joblib'\n"
            "    joblib.dump({'pipeline': pipe, 'threshold': thr, 'model_key': key}, out)\n"
            "    print(f'{dataset}: selected {key} (AUC-PR {score:.4f}) -> {out.name}')"
        ),
        md(
            "## 9. Summary\n\n"
            "- **Data preparation** — stratified 80/20 split on both datasets; "
            "features and target separated (`class` / `Class`); all scaling, "
            "encoding and resampling confined to pipeline steps refit per fold.\n"
            "- **Baseline** — Logistic Regression, evaluated on AUC-PR, F1 and the "
            "confusion matrix. Cheap, stable, and its coefficients are auditable.\n"
            "- **Ensemble** — XGBoost with a grid search over `n_estimators`, "
            "`max_depth` and `learning_rate`, plus a `scale_pos_weight` variant to "
            "test the imbalance strategy itself.\n"
            "- **Cross-validation** — 5-fold stratified, reported as mean ± std; "
            "the small std confirms the ranking is not a split artefact.\n"
            "- **Thresholds** — chosen on out-of-fold training scores, because "
            "SMOTE decalibrates the default 0.50 cut-off.\n"
            "- **Selection & justification** — see "
            "[`reports/task2-report.md`](../reports/task2-report.md).\n\n"
            "Task 3 explains the selected model with SHAP: "
            "[`shap-explainability.ipynb`](shap-explainability.ipynb)."
        ),
    ]
    build("modeling.ipynb", cells)


def shap_nb() -> None:
    cells = [
        md(
            "# Model Explainability with SHAP — Task 3\n\n"
            "**Objective.** Interpret the Task-2 selected model (XGBoost + "
            "`scale_pos_weight`) and turn the interpretation into business "
            "recommendations.\n\n"
            "**Structure**\n\n"
            "1. Built-in (gain) feature importance — top 10.\n"
            "2. SHAP global importance — beeswarm summary + mean |SHAP| bar.\n"
            "3. SHAP force / waterfall plots for a **true positive**, a **false "
            "positive** and a **false negative**.\n"
            "4. Gain vs SHAP: where the two rankings disagree, and why.\n"
            "5. Effect shape — *where* each driver flips sign, which is what makes "
            "a threshold actionable.\n"
            "6. Candidate rules scored against the labels, then recommendations.\n\n"
            "**Why both importance measures.** Gain answers *\"which features did "
            "the trees split on most profitably?\"* — global, unsigned, structural. "
            "SHAP answers *\"how much did this feature move **this** prediction, "
            "and which way?\"* Values are additive per prediction, so they support "
            "a global ranking *and* per-transaction explanations. Where the two "
            "disagree, the disagreement is itself the finding.\n\n"
            "Reusable logic lives in [`src/explainability.py`](../src/explainability.py); "
            "the batch runner that writes every figure is "
            "[`scripts/explain_models.py`](../scripts/explain_models.py)."
        ),
        code(SETUP),
        code(
            "import joblib\n"
            "import shap\n"
            "from src import modeling as md, evaluation as ev, explainability as xai\n\n"
            "print('shap', shap.__version__)\n"
            "artifact = joblib.load(config.MODELS_DIR / 'fraud_selected.joblib')\n"
            "pipe, threshold = artifact['pipeline'], artifact['threshold']\n"
            "print('model:', artifact.get('model_key'), '| threshold:', round(threshold, 4))\n"
            "print('steps:', list(dict(pipe.steps)))"
        ),
        md(
            "## 1. Rebuild the Task-2 test split\n\n"
            "Same `random_state`, so these are the exact rows the model was scored "
            "on in Task 2 — the explanations describe the reported performance, not "
            "a different sample."
        ),
        code(
            "features = list(pipe.named_steps['preprocess'].feature_names_in_)\n"
            "fraud = pd.read_csv(config.FRAUD_FEATURES)\n"
            "X, y = md.split_features_target(fraud, config.FRAUD_TARGET, features)\n"
            "for c in md.FRAUD_CATEGORICAL_COLS:\n"
            "    X[c] = X[c].astype(str)\n"
            "X_train, X_test, y_train, y_test = md.stratified_split(X, y, test_size=0.2)\n"
            "scores = ev.predict_proba_positive(pipe, X_test)\n"
            "print(f'test set: {len(X_test):,} rows, {int(y_test.sum()):,} frauds')\n"
            "pd.Series(ev.classification_metrics(y_test, scores, threshold)).to_frame('selected model').round(4)"
        ),
        md(
            "## 2. Built-in feature importance (baseline)\n\n"
            "XGBoost's `feature_importances_` uses **gain**: the average "
            "improvement in the split criterion each feature delivered. It is "
            "unsigned — a high-gain feature could push scores up, down, or both."
        ),
        code(
            "gain = xai.builtin_importance(pipe, importance_type='gain')\n"
            "weight = xai.builtin_importance(pipe, importance_type='weight')\n"
            "display(gain.head(10).rename('gain').to_frame().join(weight.rename('weight')).round(4))\n"
            "xai.plot_builtin_importance(gain, 10,\n"
            "    'Fraud_Data — XGBoost built-in importance (gain), top 10',\n"
            "    FIG / 'fraud_builtin_importance.png')\n"
            "plt.show()\n"
            "print(f'top 3 features hold {gain.head(3).sum():.1%} of total gain')"
        ),
        md(
            "## 3. SHAP — global explanation\n\n"
            "`TreeExplainer` is **exact** for tree ensembles (no sampling "
            "approximation). Values are in log-odds, additive: they sum to the "
            "model's margin minus the base value.\n\n"
            "In the beeswarm each dot is one transaction — x is that feature's "
            "SHAP value (right = pushed toward fraud), colour is the feature's own "
            "value. This is the plot that reveals whether *high* or *low* values "
            "drive fraud, which the gain bar chart cannot show."
        ),
        code(
            "expl = xai.shap_explanation(pipe, X_test)\n"
            "print('SHAP values:', expl.values.shape, '| base value (log-odds):',\n"
            "      round(float(expl.base_values[0]), 4))\n"
            "shap_imp = xai.shap_importance(expl)\n"
            "signed = xai.signed_shap_direction(expl)\n"
            "display(shap_imp.head(10).rename('mean_abs_shap').to_frame()\n"
            "        .join(signed.rename('mean_signed_shap')).round(4))"
        ),
        code(
            "# Beeswarm on a 5k subsample — 30k overlapping dots is unreadable and\n"
            "# the ranking it displays is unchanged.\n"
            "plot_expl = xai.shap_explanation(pipe, X_test, sample_size=5000)\n"
            "xai.plot_shap_summary(plot_expl, 12,\n"
            "    'Fraud_Data — SHAP summary (beeswarm)', FIG / 'fraud_shap_summary.png')\n"
            "plt.show()\n"
            "xai.plot_shap_bar(plot_expl, 12,\n"
            "    'Fraud_Data — SHAP global importance (mean |SHAP|)',\n"
            "    FIG / 'fraud_shap_bar.png')\n"
            "plt.show()"
        ),
        md(
            "## 4. Gain vs SHAP — where the rankings disagree\n\n"
            "`rank_gap` = gain rank − SHAP rank. A large positive gap means SHAP "
            "rates the feature higher than gain does: the trees rarely split on it, "
            "but when they do it moves the output."
        ),
        code(
            "comparison = xai.importance_comparison(gain, shap_imp, top_n=15)\n"
            "rho = xai.rank_correlation(gain, shap_imp)\n"
            "print(f'Spearman rho(gain, mean|SHAP|) = {rho:.4f}')\n"
            "display(comparison.round(4))\n"
            "xai.plot_importance_comparison(comparison,\n"
            "    f'Fraud_Data — gain vs mean|SHAP| (Spearman rho = {rho:.2f})',\n"
            "    FIG / 'fraud_importance_comparison.png')\n"
            "plt.show()"
        ),
        md(
            "### The redundancy that explains most of the disagreement\n\n"
            "`device_transaction_count` and `device_user_count` are ranked #1 and #2 "
            "by gain but #1 and #4 by SHAP. The check below shows why."
        ),
        code(
            "same = bool((fraud['device_transaction_count'] == fraud['device_user_count']).all())\n"
            "print('device_transaction_count == device_user_count on every row:', same)\n"
            "print('correlation:', round(float(fraud['device_transaction_count']\n"
            "      .corr(fraud['device_user_count'])), 6))\n"
            "print('\\nuser_transaction_count distinct values:',\n"
            "      fraud['user_transaction_count'].unique())\n"
            "print('\\nEvery user_id appears exactly once, so \"transactions per device\"')\n"
            "print('and \"distinct users per device\" are the same column by construction.')\n"
            "print('user_transaction_count is constant = 1: it carries zero information.')"
        ),
        md(
            "## 5. Local explanations — TP / FP / FN\n\n"
            "Extremes are chosen deliberately: the highest-scoring caught fraud "
            "(clearest win), the highest-scoring legitimate row (worst false alarm), "
            "and the lowest-scoring missed fraud (hardest miss). Feature values are "
            "shown in **original units**, not standardised ones."
        ),
        code(
            "cases = xai.select_cases(y_test, scores, threshold)\n"
            "display(xai.case_table(y_test, scores, threshold, cases))\n"
            "raw = X_test.reset_index(drop=True)"
        ),
        code(
            "for name in ['true_positive', 'false_positive', 'false_negative']:\n"
            "    pos = cases[name]\n"
            "    print('=' * 78)\n"
            "    print(f\"{name.replace('_', ' ').upper()} — fraud score {scores[pos]:.4f}\")\n"
            "    display(xai.top_contributors(expl, pos, 5).round(4))\n"
            "    print('raw values:', {k: raw.loc[pos, k] for k in\n"
            "          ['time_since_signup', 'device_transaction_count',\n"
            "           'device_velocity_24h', 'purchase_value', 'country']})\n"
            "    xai.plot_force(expl, pos, f'Force plot: {name} (score {scores[pos]:.4f})',\n"
            "                   FIG / f'fraud_force_{name}.png')\n"
            "    plt.show()\n"
            "    xai.plot_waterfall(expl, pos, 10,\n"
            "                       f'Waterfall: {name} (score {scores[pos]:.4f})',\n"
            "                       FIG / f'fraud_waterfall_{name}.png')\n"
            "    plt.show()"
        ),
        md(
            "### The instructive contrast\n\n"
            "The false positive and the true negative have almost the **same** "
            "`time_since_signup` (~7.6 h). The device feature alone decides the "
            "outcome — which is the single clearest statement of what this model "
            "has learned."
        ),
        code(
            "rows = {n: cases[n] for n in ['false_positive', 'true_negative'] if n in cases}\n"
            "display(pd.DataFrame({\n"
            "    n: {**{k: raw.loc[p, k] for k in\n"
            "            ['time_since_signup', 'device_transaction_count', 'device_velocity_24h']},\n"
            "        'fraud_score': round(float(scores[p]), 4),\n"
            "        'actual': int(y_test.to_numpy()[p])}\n"
            "    for n, p in rows.items()}).round(3))"
        ),
        md(
            "## 6. Effect shape — turning importance into thresholds\n\n"
            "A ranking says *how much* a feature matters. To act on it you need to "
            "know *where* its effect switches sign."
        ),
        code(
            "TSU_BINS = [-1, 1, 6, 24, 168, 720, 1440, 3000]  # hours: 1h/6h/1d/1w/1m/2m\n"
            "shape = xai.effect_shape(expl, 'time_since_signup', bins=TSU_BINS)\n"
            "shape['fraud_rate'] = (pd.DataFrame({'y': y_test.to_numpy(),\n"
            "        'b': pd.cut(raw['time_since_signup'], bins=TSU_BINS)})\n"
            "        .groupby('b', observed=True)['y'].mean())\n"
            "print('SHAP effect of time_since_signup, and the actual fraud rate:')\n"
            "display(shape.round(4))\n"
            "xai.plot_dependence(plot_expl, 'time_since_signup',\n"
            "    'Fraud_Data — SHAP dependence: time_since_signup',\n"
            "    FIG / 'fraud_dependence_time_since_signup.png')\n"
            "plt.show()"
        ),
        code(
            "for feat in ['device_transaction_count', 'device_velocity_24h']:\n"
            "    s = xai.effect_shape(expl, feat)\n"
            "    s['fraud_rate'] = (pd.DataFrame({'y': y_test.to_numpy(), 'v': raw[feat]})\n"
            "                       .groupby('v')['y'].mean())\n"
            "    print(f'--- {feat} ---')\n"
            "    display(s.head(8).round(4))\n"
            "    xai.plot_dependence(plot_expl, feat,\n"
            "        f'Fraud_Data — SHAP dependence: {feat}',\n"
            "        FIG / f'fraud_dependence_{feat}.png')\n"
            "    plt.show()"
        ),
        md(
            "## 7. From SHAP thresholds to scored rules\n\n"
            "Each SHAP threshold becomes a candidate rule, scored on the **full** "
            "dataset so the counts are stable. `precision` is how clean the flags "
            "are; `legit_flagged` is the customer-friction cost."
        ),
        code(
            "d, yf = fraud, fraud[config.FRAUD_TARGET]\n"
            "rules = {\n"
            "    'time_since_signup <= 1h': d['time_since_signup'] <= 1,\n"
            "    'device_velocity_24h >= 2': d['device_velocity_24h'] >= 2,\n"
            "    'device_transaction_count >= 2': d['device_transaction_count'] >= 2,\n"
            "    'device_transaction_count >= 5': d['device_transaction_count'] >= 5,\n"
            "    'tsu<=1h OR velocity>=2': (d['time_since_signup'] <= 1) | (d['device_velocity_24h'] >= 2),\n"
            "    'tsu<=1h OR device_count>=2': (d['time_since_signup'] <= 1) | (d['device_transaction_count'] >= 2),\n"
            "}\n"
            "display(xai.rule_performance(yf, rules).round(4))"
        ),
        md(
            "### What the model cannot see\n\n"
            "The rows no device/time rule reaches are the model's recall ceiling. "
            "Comparing fraud against legitimate traffic *within that region* shows "
            "whether more modelling could help."
        ),
        code(
            "uncovered = ~((d['time_since_signup'] <= 1) | (d['device_transaction_count'] >= 2))\n"
            "cols = ['time_since_signup', 'device_transaction_count',\n"
            "        'device_velocity_24h', 'purchase_value', 'age']\n"
            "print(f'{int((uncovered & (yf == 1)).sum()):,} frauds '\n"
            "      f'({(uncovered & (yf == 1)).sum() / yf.sum():.1%} of all fraud) sit here.')\n"
            "display(pd.DataFrame({\n"
            "    'fraud (uncovered)': d[uncovered & (yf == 1)][cols].median(),\n"
            "    'legitimate (uncovered)': d[uncovered & (yf == 0)][cols].median(),\n"
            "}).round(2))\n"
            "print('The two columns are indistinguishable — this residual fraud is not')\n"
            "print('reachable from these features, however the model is tuned.')"
        ),
        md(
            "## 8. creditcard — the same analysis on anonymised features\n\n"
            "The features are PCA components, so no recommendation can name a "
            "business quantity. What SHAP still gives is *which* components carry "
            "the signal, and — usefully — what the misses have in common."
        ),
        code(
            "cc_art = joblib.load(config.MODELS_DIR / 'creditcard_selected.joblib')\n"
            "cc_pipe, cc_thr = cc_art['pipeline'], cc_art['threshold']\n"
            "cc_feats = list(cc_pipe.named_steps['preprocess'].feature_names_in_)\n"
            "cc = pd.read_csv(config.CREDITCARD_CLEAN)\n"
            "Xc, yc = md.split_features_target(cc, config.CREDITCARD_TARGET, cc_feats)\n"
            "Xc_tr, Xc_te, yc_tr, yc_te = md.stratified_split(Xc, yc, test_size=0.2)\n"
            "cc_scores = ev.predict_proba_positive(cc_pipe, Xc_te)\n"
            "cc_gain = xai.builtin_importance(cc_pipe)\n"
            "cc_expl = xai.shap_explanation(cc_pipe, Xc_te, sample_size=10000)\n"
            "cc_shap = xai.shap_importance(cc_expl)\n"
            "print(f'Spearman rho = {xai.rank_correlation(cc_gain, cc_shap):.4f}')\n"
            "display(xai.importance_comparison(cc_gain, cc_shap, 10).round(4))"
        ),
        code(
            "xai.plot_shap_summary(cc_expl, 12, 'creditcard — SHAP summary (beeswarm)',\n"
            "    FIG / 'creditcard_shap_summary.png')\n"
            "plt.show()\n"
            "cc_cases = xai.select_cases(yc_te, cc_scores, cc_thr)\n"
            "display(xai.case_table(yc_te, cc_scores, cc_thr, cc_cases))"
        ),
        code(
            "# What do the missed frauds have in common? Amount is the one feature\n"
            "# here that carries business meaning.\n"
            "cc_raw = Xc_te.reset_index(drop=True)\n"
            "yt = yc_te.to_numpy(); pred = cc_scores >= cc_thr\n"
            "missed, caught = (yt == 1) & ~pred, (yt == 1) & pred\n"
            "display(pd.DataFrame({\n"
            "    'missed fraud': cc_raw['Amount'][missed].describe()[['count', '50%', 'mean']],\n"
            "    'caught fraud': cc_raw['Amount'][caught].describe()[['count', '50%', 'mean']],\n"
            "    'legitimate': cc_raw['Amount'][yt == 0].describe()[['count', '50%', 'mean']],\n"
            "}).round(2))\n"
            "f_all = cc[cc[config.CREDITCARD_TARGET] == 1]\n"
            "l_all = cc[cc[config.CREDITCARD_TARGET] == 0]\n"
            "print(f\"full data — share of fraud with Amount <= $1: {(f_all['Amount'] <= 1).mean():.1%}\")\n"
            "print(f\"full data — share of legit with Amount <= $1: {(l_all['Amount'] <= 1).mean():.1%}\")"
        ),
        md(
            "## 9. Interpretation and recommendations\n\n"
            "The written interpretation, the top-5 driver discussion, the "
            "counterintuitive findings and the full recommendation list are in "
            "**[`reports/task3-report.md`](../reports/task3-report.md)**.\n\n"
            "Headline findings:\n\n"
            "1. **Three drivers, not five.** `device_transaction_count`, "
            "`time_since_signup` and `device_velocity_24h` carry ~97% of total "
            "mean |SHAP|. Ranks 4–5 are a duplicate column and a noise-level "
            "feature.\n"
            "2. **The effect is a cliff, not a gradient.** Purchases within 1 hour "
            "of signup are 99.5% fraud; past that hour the fraud rate returns to "
            "baseline immediately.\n"
            "3. **Two engineered features were redundant or dead** — "
            "`device_user_count` is identical to `device_transaction_count`, and "
            "`user_transaction_count` is constant at 1.\n"
            "4. **The model is a smoothed rule engine.** Two hard rules reproduce "
            "its precision/recall almost exactly, which bounds what the ensemble "
            "adds on this dataset.\n"
            "5. **28% of fraud is unreachable** from these features — the uncovered "
            "region is statistically identical to legitimate traffic."
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
