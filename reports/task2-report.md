# Task 2 Report — Model Building, Training & Selection

**Project:** Fraud detection across e-commerce (`Fraud_Data.csv`) and bank
credit-card (`creditcard.csv`) transactions.

**Author:** Yostina Abera
**Date:** 2026-07-27
**Scope:** Data preparation, an interpretable baseline, a tuned ensemble,
stratified cross-validation, and a justified model selection. SHAP
explainability follows in Task 3.

---

## 1. Executive summary

Three candidates were trained on each dataset under an identical protocol and
evaluated on a held-out, **un-resampled** test set. The ensemble wins on both.

| Dataset | Selected model | Test AUC-PR | Test F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Fraud_Data** | XGBoost + `scale_pos_weight` | **0.714** | 0.690 | 0.999 | 0.527 |
| **creditcard** | XGBoost + `scale_pos_weight` | **0.822** | 0.864 | 0.987 | 0.768 |

Three findings drive the recommendation:

1. **The ensemble's margin is real, not noise.** On Fraud_Data XGBoost lifts
   AUC-PR from 0.682 to 0.723 in 5-fold CV — a **+0.040** gain against a fold
   std of **±0.006**, i.e. ~7 standard deviations. On the credit-card data the
   gap is far larger: **0.753 → 0.853 AUC-PR**.
2. **ROC-AUC would have hidden this.** On the credit-card data the logistic
   baseline scores ROC-AUC 0.981 — near-perfect-looking — while its AUC-PR is
   0.669 and its F1 at the default threshold is **0.10**. The choice of AUC-PR
   as primary metric (made in Task 1) is what surfaced the difference.
3. **`scale_pos_weight` beats SMOTE for the tree ensemble.** Reweighting the
   positive class edged out synthesising minority rows on both datasets, and did
   so without inflating the training set (227k → 453k rows for SMOTE on the
   credit-card data). SMOTE remains necessary for the linear baseline.

---

## 2. Data preparation

### 2.1 Feature/target separation
| Dataset | Rows | Features | Target | Fraud rate |
|---|---|---|---|---|
| Fraud_Data | 151,112 | 13 | `class` | 9.36% |
| creditcard | 283,726 | 30 | `Class` | 0.167% |

**Fraud_Data features** — the nine Task-1 engineered numerics
(`purchase_value`, `age`, `hour_of_day`, `day_of_week`, `time_since_signup`,
`user_transaction_count`, `device_transaction_count`, `device_user_count`,
`device_velocity_24h`) plus four categoricals (`source`, `browser`, `sex`,
`country`).

`country` was promoted from an EDA-only column in Task 1 to a model feature
here, because Task 1 measured a real spread in fraud rate across countries
(≈26% in Ecuador/Tunisia/Peru vs a 9% baseline). It has **182 levels**, so the
encoder folds levels seen in <1% of training rows into a single *infrequent*
bucket — a full one-hot expansion would add ~170 columns that are almost
entirely zero, adding variance to the linear baseline for no signal. The design
matrix ends at **34 columns**.

**creditcard features** — `Time`, `V1`–`V28`, `Amount`. These are already
anonymised PCA components, so no encoding step is needed; the pipeline is
scale → resample → model.

### 2.2 Stratified split
An 80/20 `train_test_split(..., stratify=y, random_state=42)`:

| Dataset | Train rows | Train fraud | Test rows | Test fraud |
|---|---|---|---|---|
| Fraud_Data | 120,889 | 9.365% | 30,223 | 9.364% |
| creditcard | 226,980 | 378 (0.167%) | 56,746 | 95 (0.167%) |

Stratification is not a formality on the credit-card data: with 473 positives
in total, an unstratified split can hand one half a materially different
positive rate, which would make the test metrics incomparable to production
prevalence.

### 2.3 Leakage control — the design decision that matters most
Scaling, encoding **and** SMOTE are steps inside a single `imblearn` pipeline
([`src/modeling.py`](../src/modeling.py)), not preprocessing applied up front:

```
Pipeline([("preprocess", ColumnTransformer), ("smote", SMOTE), ("clf", estimator)])
```

`imblearn` invokes samplers on `fit` only — never on `transform`/`predict`. Two
consequences:

- Every cross-validation fold and every grid-search candidate refits its own
  scaler, encoder and SMOTE on that fold's training portion. Resampling before
  cross-validating is the standard way this task is silently corrupted: synthetic
  minority points interpolated from a row that later lands in the validation fold
  leak the label, and AUC-PR is inflated.
- The test set is scored at its **real 9.36% / 0.167% prevalence**. Nothing
  synthetic ever reaches an evaluated row.

A regression test locks this in: `test_smote_does_not_change_prediction_row_count`
fails if resampling ever escapes into the transform path.

---

## 3. Models

| Key | Model | Imbalance strategy | Role |
|---|---|---|---|
| `logreg_smote` | Logistic Regression (L2, lbfgs) | SMOTE | interpretable baseline |
| `xgb_smote` | XGBoost (hist) | SMOTE | ensemble |
| `xgb_weighted` | XGBoost (hist) | `scale_pos_weight = n_neg/n_pos` | ensemble |

The third candidate exists so the **imbalance strategy** is measured rather than
assumed. SMOTE synthesises minority rows; `scale_pos_weight` multiplies the
gradient contribution of positives (10.7 on Fraud_Data, 599.5 on the credit-card
data). Holding the model family and split fixed isolates that one difference.

**Why XGBoost for the ensemble.** Fraud in this data is an *interaction*: Task 1
showed fraud sits at `time_since_signup` ≈ 0 hours **and** on devices shared by
~7 accounts. A linear model can only add those effects; boosted trees can
represent the conjunction. Gradient boosting also handles the mixed scales and
skew of the count features without further transformation, and XGBoost's
`aucpr` objective optimises the metric being reported.

### 3.1 Hyperparameter tuning
`GridSearchCV`, 3-fold stratified, scored by `average_precision` (AUC-PR).
3-fold rather than 5 for the search — it only has to rank candidates, and the
winner is then re-validated with the full 5-fold protocol.

| Dataset | Model | Grid | Winner | CV AUC-PR |
|---|---|---|---|---|
| Fraud_Data | `logreg_smote` | `C ∈ {0.1, 1, 10}` | `C=0.1` | 0.683 ± 0.011 |
| Fraud_Data | `xgb_smote` | `n_estimators ∈ {200,400}` × `max_depth ∈ {3,6}` × `lr ∈ {0.05,0.2}` | `400, depth 3, lr 0.2` | 0.721 ± 0.004 |
| Fraud_Data | `xgb_weighted` | same | `200, depth 3, lr 0.05` | **0.723 ± 0.005** |
| creditcard | `logreg_smote` | `C ∈ {0.1, 1, 10}` | `C=0.1` | 0.753 ± 0.024 |
| creditcard | `xgb_smote` | same 8-point grid | `400, depth 6, lr 0.2` | 0.852 ± 0.022 |
| creditcard | `xgb_weighted` | same | `400, depth 6, lr 0.05` | **0.854 ± 0.020** |

Both datasets chose the **strongest regularisation available** for the baseline
(`C=0.1`), which is the expected response to a SMOTE-expanded training set. The
depth preference splits by dataset: **depth 3** on Fraud_Data (13 features, the
signal is a couple of strong interactions) versus **depth 6** on the credit-card
data (30 decorrelated PCA components, deeper trees have more to combine).

Full grids: [`task2_tuning_fraud.csv`](task2_tuning_fraud.csv),
[`task2_tuning_creditcard.csv`](task2_tuning_creditcard.csv).

### 3.2 Threshold selection
The default 0.50 cut-off is not meaningful for a model trained on rebalanced
data: SMOTE calibrates scores to a 50/50 world, so 0.50 systematically
over-predicts fraud. Thresholds were therefore chosen by maximising F1 on
**out-of-fold training** probabilities (`cross_val_predict`, 3-fold) — no test
data involved. Both 0.50 and the tuned threshold are reported below so the effect
is visible.

---

## 4. Results

### 4.1 Fraud_Data — 5-fold stratified CV (training half)
| Model | AUC-PR | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression + SMOTE | 0.6823 ± 0.0073 | 0.6247 ± 0.0027 | 0.5870 ± 0.0065 | 0.6676 ± 0.0049 | 0.8437 ± 0.0064 |
| XGBoost + SMOTE | 0.7212 ± 0.0061 | 0.7009 ± 0.0061 | 0.9950 ± 0.0014 | 0.5410 ± 0.0071 | 0.8452 ± 0.0034 |
| **XGBoost + `scale_pos_weight` (10.7)** | **0.7226 ± 0.0059** | 0.6189 ± 0.0034 | 0.5460 ± 0.0034 | **0.7142 ± 0.0059** | **0.8463 ± 0.0041** |

### 4.2 Fraud_Data — held-out test set (30,223 rows, 2,830 frauds)
| Model | Threshold | AUC-PR | F1 | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| XGBoost + `scale_pos_weight` | 0.50 | 0.7141 | 0.6133 | 0.5388 | **0.7117** | 2,014 | 1,724 | 816 |
| XGBoost + `scale_pos_weight` | 0.896 | **0.7141** | **0.6900** | 0.9993 | 0.5269 | 1,491 | **1** | 1,339 |
| XGBoost + SMOTE | 0.453 | 0.7118 | 0.6891 | 0.9811 | 0.5311 | 1,503 | 29 | 1,327 |
| Logistic Regression + SMOTE | 0.806 | 0.6750 | 0.6677 | 0.9061 | 0.5286 | 1,496 | 155 | 1,334 |
| Logistic Regression + SMOTE | 0.50 | 0.6750 | 0.6210 | 0.5849 | 0.6618 | 1,873 | 1,329 | 957 |

*Figures: `figures/fraud_pr_curves.png`, `figures/fraud_confusion_matrices.png`.*

The two rows for the winning model are the same trained model at two operating
points, and they expose the real decision: **1,724 false positives for 523 extra
frauds caught**, or near-zero false positives at 53% recall. AUC-PR — which is
threshold-free — is identical for both. That is why AUC-PR is the selection
metric and the threshold is a separate, business-owned parameter (§6).

### 4.3 creditcard — 5-fold stratified CV (training half)
| Model | AUC-PR | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression + SMOTE | 0.7529 ± 0.0262 | 0.1039 ± 0.0097 | 0.0551 ± 0.0055 | **0.9073 ± 0.0337** | 0.9809 ± 0.0082 |
| XGBoost + SMOTE | 0.8486 ± 0.0314 | 0.8162 ± 0.0249 | 0.8151 ± 0.0273 | 0.8200 ± 0.0510 | 0.9780 ± 0.0053 |
| **XGBoost + `scale_pos_weight` (599.5)** | **0.8531 ± 0.0271** | **0.8640 ± 0.0260** | **0.9092 ± 0.0369** | 0.8254 ± 0.0453 | **0.9820 ± 0.0057** |

Fold std is ~4× wider here than on Fraud_Data (±0.027 vs ±0.006) — with only
~75 frauds per fold, that is the irreducible cost of 473 positives, and it is
exactly why the ranking is taken from CV rather than a single split.

### 4.4 creditcard — held-out test set (56,746 rows, 95 frauds)
| Model | Threshold | AUC-PR | F1 | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| XGBoost + `scale_pos_weight` | 0.851 | **0.8224** | **0.8639** | 0.9865 | 0.7684 | 73 | **1** | 22 |
| XGBoost + `scale_pos_weight` | 0.50 | 0.8224 | 0.8621 | 0.9494 | 0.7895 | 75 | 4 | 20 |
| XGBoost + SMOTE | 0.961 | 0.8190 | 0.8383 | 0.9722 | 0.7368 | 70 | 2 | 25 |
| XGBoost + SMOTE | 0.50 | 0.8190 | 0.8128 | 0.8261 | 0.8000 | 76 | 16 | 19 |
| Logistic Regression + SMOTE | 1.000 | 0.6687 | 0.8045 | 0.8571 | 0.7579 | 72 | 12 | 23 |
| Logistic Regression + SMOTE | 0.50 | 0.6687 | **0.1001** | 0.0531 | 0.8737 | 83 | **1,481** | 12 |

*Figures: `figures/creditcard_pr_curves.png`,
`figures/creditcard_confusion_matrices.png`.*

The baseline's last row is the clearest single result in this task. At the
default threshold it flags **1,481 legitimate transactions to catch 83 frauds** —
precision 5.3%, F1 0.10 — while simultaneously reporting ROC-AUC **0.963**. Any
evaluation built on ROC-AUC or accuracy would have called this model excellent.

That row also exposes a fragility the metrics table alone doesn't show: the
baseline is only usable at a threshold of **1.000**. SMOTE at a 599:1 imbalance
pushes ~1,500 test transactions to a probability that rounds to 1.0, so the
entire precision/recall trade-off is compressed into the last floating-point
increment of the score range. The tuned threshold recovers F1 0.80, but an
operating point that sits on the numerical boundary is not something to deploy —
a small distribution shift moves it arbitrarily.

---

## 5. Model selection

**Selected: XGBoost with `scale_pos_weight`, for both datasets.**
Artifacts: `models/fraud_xgb_weighted.joblib`,
`models/creditcard_xgb_weighted.joblib` (each stores the fitted pipeline, its
tuned threshold, and the feature list).

### 5.1 Justification

**Performance.** Highest AUC-PR on both datasets in 5-fold CV *and* on the test
set — the only candidate to lead on both. The margin over the baseline is
+0.040 AUC-PR on Fraud_Data (±0.006 fold std) and +0.100 on the credit-card data
(±0.027). It also leads on F1, precision and even ROC-AUC on the credit-card
data, so nothing is being traded away for the AUC-PR gain.

**Error profile.** At its tuned threshold on Fraud_Data it produces **1 false
positive** across 27,393 legitimate transactions while catching 53% of fraud, and
the *same model* reaches 71% recall at 0.50. The baseline cannot reach either
corner: at 0.50 it emits 1,329 false positives for 66% recall, and at its tuned
threshold it still emits 155 for lower recall than the ensemble. The ensemble
dominates the baseline's whole operating curve, which is what the PR-curve figure
shows directly.

**Imbalance strategy.** `scale_pos_weight` beat SMOTE on both datasets (+0.001
and +0.005 AUC-PR — small, but consistent, and with tighter CV variance on the
credit-card data). It is also the cheaper and more robust choice: no synthetic
rows means the credit-card training set stays at 227k instead of 453k (its grid
search ran in **222 s versus 409 s**), and nothing depends on SMOTE's
nearest-neighbour interpolation
being sensible in a 30-dimensional space where the minority class has 378
examples. **SMOTE is still required for the linear baseline** — without
rebalancing, logistic regression on 0.167% positives collapses toward the
majority class.

**Interpretability — the genuine trade-off.** Logistic regression is the more
interpretable model: each coefficient is a signed log-odds contribution, readable
without tooling. Three reasons that does not outweigh the performance gap here:

1. **The gap is too large to concede**, especially on the credit-card data, where
   the baseline's usable operating point sits at a threshold of 1.000. An
   interpretable model that cannot be given a stable threshold is not a
   deployable model.
2. **The credit-card features are PCA components.** `V14`'s coefficient is not
   interpretable in any business sense — the anonymisation already destroyed what
   made the linear model auditable. Interpretability is nearly free to give up on
   this dataset.
3. **The gap is closable with tooling.** SHAP (Task 3) gives the XGBoost model
   both global feature importance and per-transaction attributions, which is
   strictly more than a coefficient vector: it explains *individual* decisions,
   which is what a fraud analyst reviewing a flagged transaction actually needs.

The baseline earns its place regardless: it set the bar, its full grid search
costs 25 s against the ensemble's 233 s, and it stays in the repo as the
reference any future model must beat.

### 5.2 What was not chosen, and why
- **XGBoost + SMOTE** — statistically indistinguishable from the weighted
  variant, but pays for it in training cost and in dependence on synthetic
  minority points. Kept as a trained artifact for comparison.
- **Random Forest / LightGBM** — the task called for one ensemble; XGBoost was
  picked for its native `aucpr` objective and `scale_pos_weight`. LightGBM is not
  installed in this environment, which would have added an unpinned dependency
  for no expected gain.
- **Accuracy as a metric** — excluded throughout. A constant "legitimate"
  prediction scores 90.6% / 99.83% and catches zero fraud.

---

## 6. Deployment note — the threshold is a business parameter

The selected model's recall/precision trade-off on Fraud_Data spans **53% recall
at 1 false positive** to **71% recall at 1,724 false positives**, from one
trained model. The reported threshold maximises F1, which weights a missed fraud
and a blocked customer equally. That is a placeholder, not a recommendation: the
right cut-off follows from the ratio of average fraud loss to the cost of a
declined legitimate transaction. Once that ratio is known, re-running
`ev.best_f1_threshold` against a cost-weighted objective on out-of-fold training
scores picks the operating point without retraining.

Two limits worth stating plainly:

- **Recall on Fraud_Data plateaus around 71%.** No candidate exceeded it. The
  residual ~29% of fraud does not look automated (no near-zero
  `time_since_signup`, no shared device), so it is likely not reachable from
  these features. Closing it needs new signal — payment-instrument history,
  shipping-address mismatch — not more tuning.
- **The credit-card estimates rest on 95 test frauds.** A ±0.027 CV std is the
  honest precision of the AUC-PR figures. The ranking is stable across folds; the
  third decimal place is not.

---

## 7. Reproducibility

```bash
pytest                                       # 55 tests, incl. leakage guards
python scripts/train_models.py               # full protocol, both datasets (~26 min)
python scripts/train_models.py --quick       # 10% sample smoke test (~2 min)
python scripts/build_notebooks.py            # regenerate notebooks
jupyter nbconvert --to notebook --execute --inplace notebooks/modeling.ipynb
```

`random_state=42` is fixed for the split, SMOTE, both estimators and every CV
splitter.

| Artifact | Path |
|---|---|
| Trained pipelines + thresholds | `models/{fraud,creditcard}_{logreg_smote,xgb_smote,xgb_weighted}.joblib` |
| Selected model per dataset | `models/{fraud,creditcard}_selected.joblib` |
| Test metrics | `reports/task2_test_metrics_{fraud,creditcard}.csv` |
| 5-fold CV metrics | `reports/task2_cv_metrics_{fraud,creditcard}.csv` |
| Grid-search results | `reports/task2_tuning_{fraud,creditcard}.csv` |
| Machine-readable summary | `reports/task2_summary.json` |
| Figures | `reports/figures/{fraud,creditcard}_{pr_curves,confusion_matrices}.png` |
| Narrative | `notebooks/modeling.ipynb` |
| Code | `src/modeling.py`, `src/evaluation.py`, `scripts/train_models.py` |

---

## 8. Next — Task 3

SHAP on `models/{fraud,creditcard}_xgb_weighted.joblib`: a global summary plot to
confirm whether `time_since_signup` and `device_user_count` dominate as Task 1's
EDA predicted, and waterfall plots on individual flagged transactions to give
analysts a per-decision explanation. This is also the check on whether the
ensemble's advantage comes from the interaction between those two features, as
argued in §3.
