# Task 3 Report — Model Explainability

**Project:** Fraud detection across e-commerce (`Fraud_Data.csv`) and bank
credit-card (`creditcard.csv`) transactions.

**Author:** Yostina Abera
**Date:** 2026-07-27
**Model explained:** XGBoost + `scale_pos_weight`, the Task-2 selection
(`models/{fraud,creditcard}_selected.joblib`), on the identical Task-2 test split.

---

## 1. Executive summary

SHAP confirms Task 1's behavioural hypothesis and then sharpens it into numbers
you can act on. Five findings, in descending order of consequence:

1. **There are three drivers, not five.** `device_transaction_count`,
   `time_since_signup` and `device_velocity_24h` hold **97.3%** of total
   mean |SHAP|. Ranks 4 and 5 are a duplicate column and a noise-level feature.
2. **The signal is a cliff, not a gradient.** Purchases made within **1 hour** of
   signup are **99.52% fraud** (7,604 of 7,641). Past that hour the fraud rate
   drops to ~4.7% — below the 9.36% base rate. Within **1 minute** it is
   **100.0%** across 7,600 transactions.
3. **Two of Task 1's nine engineered numerics are redundant or dead.**
   `device_user_count` is *identical* to `device_transaction_count` on every row
   (r = 1.000000), and `user_transaction_count` is the constant 1 for all 151,112
   rows. This is why gain and SHAP disagree so sharply (Spearman ρ = 0.42).
4. **The model is a smoothed rule engine.** Two hard rules reproduce its
   precision and recall almost exactly. That bounds what the ensemble adds on
   *this* dataset — and it is an argument for deploying rules *and* the model for
   different purposes, not one instead of the other.
5. **28.3% of fraud is unreachable from these features.** In the region no
   device/time rule touches, fraudulent and legitimate transactions are
   statistically indistinguishable (median `time_since_signup` 1,463 h vs
   1,444 h; identical device counts, purchase values and ages). This is the
   precise, quantified explanation of the ~71% recall ceiling found in Task 2.

---

## 2. Feature importance baseline (built-in gain)

XGBoost's `feature_importances_` reports **gain** — the average improvement in
the split criterion each feature delivered. It is global and unsigned.

### 2.1 Fraud_Data — top 10 by gain
| Rank | Feature | Gain | Weight (split count) |
|---|---|---|---|
| 1 | `device_transaction_count` | 0.5229 | 0.1949 |
| 2 | `device_user_count` | 0.1707 | 0.0285 |
| 3 | `time_since_signup` | 0.1546 | 0.2203 |
| 4 | `device_velocity_24h` | 0.0195 | 0.0639 |
| 5 | `source_Direct` | 0.0177 | 0.0277 |
| 6 | `browser_IE` | 0.0064 | 0.0031 |
| 7 | `country_Italy` | 0.0058 | 0.0054 |
| 8 | `browser_Chrome` | 0.0055 | 0.0054 |
| 9 | `browser_Safari` | 0.0054 | 0.0046 |
| 10 | `browser_Opera` | 0.0053 | 0.0008 |

The top three hold **84.8%** of total gain. *Figure:
`figures/fraud_builtin_importance.png`.*

### 2.2 creditcard — top 10 by gain
| Rank | Feature | Gain | | Rank | Feature | Gain |
|---|---|---|---|---|---|---|
| 1 | `V14` | 0.5035 | | 6 | `V19` | 0.0243 |
| 2 | `V10` | 0.0811 | | 7 | `V3` | 0.0221 |
| 3 | `V4` | 0.0603 | | 8 | `V13` | 0.0192 |
| 4 | `V12` | 0.0353 | | 9 | `V17` | 0.0172 |
| 5 | `V8` | 0.0298 | | 10 | `V11` | 0.0157 |

`V14` alone takes half the total gain. *Figure:
`figures/creditcard_builtin_importance.png`.*

---

## 3. SHAP analysis

`TreeExplainer` — **exact** for tree ensembles, no sampling approximation.
Computed on the full test set (30,223 / 56,746 rows); values are in log-odds and
additive (they sum to the model margin minus the base value, a property asserted
in `test_shap_values_sum_to_model_margin`).

### 3.1 Global summary — Fraud_Data
| Rank | Feature | mean \|SHAP\| | mean signed SHAP | Share of total |
|---|---|---|---|---|
| 1 | `device_transaction_count` | 0.9358 | −0.6005 | 58.6% |
| 2 | `time_since_signup` | 0.4322 | −0.0195 | 27.1% |
| 3 | `device_velocity_24h` | 0.1136 | +0.0853 | 7.1% |
| 4 | `device_user_count` | 0.0776 | −0.0506 | 4.9% |
| 5 | `purchase_value` | 0.0319 | −0.0032 | 2.0% |
| 6 | `source_Direct` | 0.0267 | −0.0034 | 1.7% |
| 7 | `day_of_week` | 0.0255 | −0.0019 | 1.6% |
| 8 | `age` | 0.0230 | −0.0026 | 1.4% |
| 9 | `hour_of_day` | 0.0177 | −0.0012 | 1.1% |
| 10 | `source_SEO` | 0.0142 | −0.0002 | 0.9% |

*Figures: `figures/fraud_shap_summary.png` (beeswarm),
`figures/fraud_shap_bar.png`.*

Two things the beeswarm shows that the table cannot. First, the direction: the
`device_transaction_count` dots at SHAP ≈ −0.9 are all **blue** (low value) and
the dots at +1.1 to +1.7 are **red** (high value) — more transactions per device
means more fraud, monotonically. Second, the *shape* of `time_since_signup`: a
tight isolated cluster at SHAP ≈ +4.0 (blue = near-zero hours) and everything
else compressed around −0.24. That bimodality is the cliff quantified in §3.3.

### 3.2 Global summary — creditcard
| Rank | Feature | mean \|SHAP\| | mean signed SHAP |
|---|---|---|---|
| 1 | `V14` | 2.5346 | −2.4352 |
| 2 | `V4` | 1.7061 | −1.5081 |
| 3 | `V12` | 1.0529 | −0.9976 |
| 4 | `V10` | 0.9001 | −0.8456 |
| 5 | `V11` | 0.8022 | −0.7722 |
| 6 | `V3` | 0.7380 | −0.7214 |
| 7 | `V8` | 0.5153 | −0.4406 |
| 8 | `V19` | 0.4519 | −0.3962 |

Every top feature has mean signed SHAP ≈ −mean |SHAP|, i.e. it almost always
pushes *toward legitimate*. That is what a 0.167% base rate looks like in SHAP
space: the model's default posture is "legitimate", and fraud detection consists
of a handful of components firing hard in the opposite direction on 473 rows.

Task 1's EDA flagged `V14, V12, V17, V10` as the strongest target correlates.
SHAP confirms V14, V12 and V10, but **promotes `V4` to rank 2** (gain rank 3,
Task-1 correlation rank ~5) and **demotes `V17` to rank 16**. Linear correlation
missed V4's contribution because it is conditional, not marginal.

### 3.3 Effect shape — where each driver flips sign

A ranking says how much a feature matters. This says *where* — and it is the step
that makes a threshold defensible rather than arbitrary. Fraud rate is measured
on the test set.

**`time_since_signup`** (*figure: `figures/fraud_dependence_time_since_signup.png`*)
| Hours since signup | n | mean SHAP | Direction | Fraud rate |
|---|---|---|---|---|
| **≤ 1 h** | 1,500 | **+4.0370** | toward fraud | **99.53%** |
| 1–6 h | 59 | +1.3373 | toward fraud | 3.39% |
| 6–24 h | 186 | +0.4004 | toward fraud | 4.84% |
| 1–7 d | 1,406 | −0.2748 | toward legit | 5.48% |
| 7–30 d | 5,600 | −0.2534 | toward legit | 4.59% |
| 30–60 d | 7,158 | −0.2235 | toward legit | 4.40% |
| > 60 d | 14,314 | −0.2370 | toward legit | 4.73% |

The effect collapses by a factor of **three within the first hour** and by
**ten by hour six**. There is no gradual decay — a single hard boundary.

**`device_transaction_count`** (*figure:
`figures/fraud_dependence_device_transaction_count.png`*)
| Transactions on device | n | mean SHAP | Fraud rate |
|---|---|---|---|
| 1 | 26,332 | −0.8813 | **3.07%** |
| 2 | 2,188 | +1.1095 | **23.40%** |
| 3 | 52 | +1.4856 | 25.00% |
| 5 | 17 | +1.6123 | 70.59% |
| 6–12 | 727 | +1.41 to +1.56 | **86–90%** |

One repeat on a device multiplies the fraud rate by **7.6×**. Five or more and
it is ~88%.

**`device_velocity_24h`** — the sharpest boundary in the data
| Purchases from device in 24 h | n | mean SHAP | Fraud rate |
|---|---|---|---|
| 1 | 28,864 | −0.0148 | **5.13%** |
| 2 | 176 | +0.8056 | **94.89%** |
| ≥ 3 | 1,183 | +2.46 | **99.9–100%** |

---

## 4. Local explanations — three individual predictions

Cases were chosen as extremes rather than at random: the highest-scoring caught
fraud, the highest-scoring false alarm, and the lowest-scoring miss. Feature
values below are in **original units** (the plots substitute unscaled values for
display; attributions are untouched).

| Case | Row | Actual | Fraud score | Predicted |
|---|---|---|---|---|
| True positive | 21,684 | fraud | 0.999784 | fraud |
| False positive | 26,610 | legitimate | 0.999668 | fraud |
| False negative | 11,806 | fraud | 0.076418 | legitimate |
| True negative | 11,844 | legitimate | 0.041028 | legitimate |

*Figures: `figures/fraud_{force,waterfall}_{true_positive,false_positive,false_negative,true_negative}.png`.*

### 4.1 True positive — textbook automated fraud
| Feature | Value | SHAP | Direction |
|---|---|---|---|
| `time_since_signup` | **0.00028 h (≈ 1 second)** | **+4.041** | toward fraud |
| `device_velocity_24h` | 7 | +2.223 | toward fraud |
| `device_transaction_count` | **20** | +1.924 | toward fraud |
| `device_user_count` | 20 | +0.162 | toward fraud |
| `hour_of_day` | 6 | +0.028 | toward fraud |

Base value 0.012 → margin **8.44** log-odds. A purchase one second after signup,
on a device already carrying 20 accounts, with 7 purchases in the last 24 hours.
Three features supply 96% of the push. Note `purchase_value` = $47 contributes
+0.02 — essentially nothing. The model is not detecting an unusual *purchase*; it
is detecting an unusual *account and device*.

### 4.2 False positive — the model is arguably right and the label is the anomaly
| Feature | Value | SHAP | Direction |
|---|---|---|---|
| `time_since_signup` | 7.62 h | +3.917 | toward fraud |
| `device_velocity_24h` | **7** | +2.345 | toward fraud |
| `device_transaction_count` | **7** | +1.590 | toward fraud |
| `source_Direct` | 1 | +0.075 | toward fraud |
| `purchase_value` | $21 | −0.027 | toward legit |

This transaction has the *same* signature as the true positive: 7 accounts on one
device, 7 purchases in 24 hours. Nothing about the model's reasoning is faulty —
it fired on a pattern that is 86–90% fraud. **At the deployed threshold this is
the only false positive among 27,393 legitimate test transactions**, so the
model's error rate here is not a tuning problem to fix but a residual ambiguity
in the data (a genuinely shared household or office device, or an unlabelled
fraud).

### 4.3 False negative — fraud invisible to a device-centric model
| Feature | Value | SHAP | Direction |
|---|---|---|---|
| `device_transaction_count` | **1** | **−2.934** | toward legit |
| `time_since_signup` | 4.78 h | +1.114 | toward fraud |
| `device_user_count` | 1 | −0.396 | toward legit |
| `purchase_value` | $18 | −0.149 | toward legit |
| `country_infrequent` | 0 | −0.106 | toward legit |

The miss is fully explained: `time_since_signup` = 4.78 h correctly pushed
+1.11 toward fraud, but the solo device pushed **−2.93** and overwhelmed it. This
is the failure mode of a device-centric model — **fraud on a fresh, unshared
device, executed outside the one-hour window, is invisible.**

### 4.4 The decisive contrast
The false positive and the true negative have nearly identical
`time_since_signup` — 7.62 h and 7.64 h — and opposite outcomes:

| | `time_since_signup` | `device_transaction_count` | `device_velocity_24h` | Score | Actual |
|---|---|---|---|---|---|
| False positive | 7.62 h | **7** | **7** | 0.9997 | legitimate |
| True negative | 7.64 h | **1** | **1** | 0.0410 | legitimate |

Same account age, opposite verdicts, decided entirely by the device features.
This is the clearest single statement of what the model has learned.

### 4.5 creditcard — same structure, anonymised
| Case | Top contributions | Reading |
|---|---|---|
| True positive (score 1.0000) | `V14` = −14.35 (**+4.797**), `V10` = −11.93 (+2.090), `V12` = −13.16 (+1.987), `V4` = +6.30 (+1.380) | extreme excursion on all four |
| False positive (score 1.0000) | `V14` = −7.45 (**+4.937**), `V10` = −5.24 (+2.051), `V12` = −7.16 (+1.693), `V4` = +2.69 (+1.162) | the **same signature at half the magnitude** |
| False negative (score 0.0000) | `V14` = −0.007 (**−2.115**), `V12` = +0.69 (−1.525), `V11` = −0.86 (−1.333) | every component sits at its normal value |

The false positive is not a different pattern — it is a milder version of the
true positive, which is exactly the ambiguity a single threshold cannot resolve.
The false negative left **no trace at all** in PCA space: `V14` = −0.007 against
the fraud-typical −14. Its `Amount` was **$1.18**.

---

## 5. Comparing SHAP with built-in importance

| Dataset | Spearman ρ (gain, mean \|SHAP\|) |
|---|---|
| Fraud_Data | **0.4175** |
| creditcard | **0.5573** |

Neither is close to 1 — the two measures genuinely disagree, and each
disagreement has a cause. *Figures:
`figures/{fraud,creditcard}_importance_comparison.png`.*

### 5.1 Fraud_Data — the largest gaps
| Feature | Gain | mean \|SHAP\| | Gain rank | SHAP rank | Rank gap |
|---|---|---|---|---|---|
| `device_transaction_count` | 0.5229 | 0.9358 | 1 | 1 | 0 |
| `time_since_signup` | 0.1546 | 0.4322 | 3 | 2 | +1 |
| `device_velocity_24h` | 0.0195 | 0.1136 | 4 | 3 | +1 |
| **`device_user_count`** | 0.1707 | 0.0776 | **2** | **4** | **−2** |
| **`purchase_value`** | 0.0041 | 0.0319 | **22** | **5** | **+17** |
| `source_SEO` | 0.0031 | 0.0142 | 29 | 10 | +19 |
| `hour_of_day` | 0.0035 | 0.0177 | 24 | 9 | +15 |
| `age` | 0.0042 | 0.0230 | 21 | 8 | +13 |

Three distinct explanations, and it matters which is which:

**`device_user_count` (gain #2 → SHAP #4): perfect redundancy.** It is *identical*
to `device_transaction_count` on all 151,112 rows. Gain credits whichever
duplicate a tree happened to split on, inflating the pair's combined apparent
importance; SHAP splits credit between perfectly-correlated features, so the
duplicate's share collapses. **SHAP is right and gain is misleading here** — the
column carries no independent information at all.

**`purchase_value` (gain #22 → SHAP #5): rarely split on, but moves the output
when it is.** Low gain means the trees seldom found a profitable split;
non-trivial mean |SHAP| means those few splits shift predictions.

**The flat tail: don't over-read ranks 5–10.** `purchase_value`'s mean |SHAP| is
0.0319 — **29× smaller** than `device_transaction_count`'s 0.9358. Its class
means are effectively identical (fraud $36.99 vs legitimate $36.93), exactly as
Task 1 found. Being "5th most important" out of a set where the top 3 hold 97.3%
of the total is not evidence of a real effect. Reported here as a caution: a
top-N SHAP list read without magnitudes invites exactly this mistake.

### 5.2 creditcard
Gain concentrates **50.4%** of importance on `V14`; SHAP spreads across V14
(2.53), V4 (1.71), V12 (1.05), V10 (0.90) and V11 (0.80). `V5`, `V1` and `V28`
show rank gaps of +16, +18 and +13 — split on rarely, influential when they are.
Depth-6 trees make many such conditional splits, which is precisely the
interaction structure gain cannot represent.

---

## 6. Top 5 drivers of fraud predictions

Ranked by mean |SHAP|, with the honest caveat attached:

| # | Driver | mean \|SHAP\| | Share | Direction | Verdict |
|---|---|---|---|---|---|
| 1 | `device_transaction_count` | 0.9358 | 58.6% | ↑ with count; flips at **2** | **Real, dominant** |
| 2 | `time_since_signup` | 0.4322 | 27.1% | ↓ with hours; cliff at **1 h** | **Real, decisive** |
| 3 | `device_velocity_24h` | 0.1136 | 7.1% | ↑ with count; flips at **2** | **Real, sharpest boundary** |
| 4 | `device_user_count` | 0.0776 | 4.9% | same as #1 | **Duplicate of #1** — not independent |
| 5 | `purchase_value` | 0.0319 | 2.0% | mixed, near-zero mean | **Noise-level** — 29× below #1 |

**The defensible answer is that there are three drivers.** Ranks 1–3 hold
**97.3%** of total mean |SHAP| and each has a clean, monotone, threshold-shaped
effect. Rank 4 is the same column as rank 1 under a different name. Rank 5 has no
marginal separation between classes. Reporting "five drivers" would be padding a
list to fit the question.

All three are **behavioural** — how the account and device behave. None is
demographic (`age` rank 8, `sex` unranked) and none concerns the purchase itself
(`purchase_value` rank 5 at noise level). This confirms Task 1's central claim and
justifies the feature-engineering effort: the raw columns carried nearly no
signal, and every driver the model uses was constructed in Task 1.

---

## 7. Surprising and counterintuitive findings

**7.1 Two of nine engineered numerics were redundant or dead.**
`device_user_count` ≡ `device_transaction_count` on every row, and
`user_transaction_count` is the constant 1 for all 151,112 rows. The cause is a
data-structure property invisible in Task 1's univariate EDA: **every `user_id`
appears exactly once** (151,112 unique users, 151,112 rows). So "transactions per
user" is always 1, and "distinct users per device" is by construction identical
to "transactions per device". Task 1 reported the pair as two independent
findings (`device_user_count` averaging 7.15 for fraud vs 1.12) — the statistic
is correct, but it is one finding, not two. **Both columns should be dropped**:
they add dimensionality, split gain three ways, and inflate apparent importance.

**7.2 The strongest feature is not the one Task 1 headlined.** Task 1 named
`time_since_signup` the strongest single signal. SHAP ranks
`device_transaction_count` first, at **2.2× the magnitude**. Both are right about
their own question: `time_since_signup` ≤ 1 h has near-perfect *precision*
(99.5%), but it only covers 5% of rows, so its *average* contribution across the
test set is lower. Precision on a narrow slice and average influence across the
population are different quantities, and a SHAP ranking measures the second.

**7.3 The model is, functionally, a smoothed two-rule engine.** Scored on the
full dataset:

| Rule / model | Flagged | Frauds caught | Recall | Precision | Legit flagged |
|---|---|---|---|---|---|
| `time_since_signup ≤ 1h` | 7,641 | 7,604 | 53.7% | **99.52%** | **37** |
| `device_velocity_24h ≥ 2` | 6,903 | 6,854 | 48.4% | 99.29% | 49 |
| `device_transaction_count ≥ 5` | 8,391 | 7,625 | 53.9% | 90.87% | 766 |
| `tsu ≤ 1h OR velocity ≥ 2` | 7,704 | 7,618 | 53.8% | 98.88% | 86 |
| `device_transaction_count ≥ 2` | 19,331 | 10,141 | **71.7%** | 52.46% | 9,190 |
| — *XGBoost @ threshold 0.896* | — | — | 52.7% | 99.93% | 1 (of 27,393) |
| — *XGBoost @ threshold 0.50* | — | — | 71.2% | 53.88% | 1,724 |

The correspondence is near-exact at both operating points: the two-rule union
matches the model's tuned operating point (53.8% / 98.9% vs 52.7% / 99.9%), and
`device_transaction_count ≥ 2` matches its 0.50 point (71.7% / 52.5% vs
71.2% / 53.9%). This does not make the model worthless — it produces calibrated
scores, degrades gracefully when patterns drift, needs no manual re-derivation of
thresholds, and generalises to feature combinations no analyst enumerated. But it
does mean **the honest claim is "the ensemble matches a two-rule baseline on this
dataset", not "the ensemble found something rules cannot"**. The rules should ship
as a fast deterministic pre-filter and the model as the scoring layer behind it.

**7.4 The recall ceiling is a property of the data, not the model.** No Task-2
candidate exceeded ~71% recall. `device_transaction_count ≥ 2` catches 71.7% — the
same number. The remaining **4,008 frauds (28.3%)** sit in a region where they
are indistinguishable from legitimate traffic:

| Median | Fraud (uncovered) | Legitimate (uncovered) |
|---|---|---|
| `time_since_signup` | 1,463 h | 1,444 h |
| `device_transaction_count` | 1 | 1 |
| `device_velocity_24h` | 1 | 1 |
| `purchase_value` | $34 | $35 |
| `age` | 32 | 33 |

No threshold, resampling scheme or architecture separates these populations —
the information is not in the features. Closing that gap requires **new signal**,
not more tuning.

**7.5 On the credit-card model, the missed frauds skew small.** Missed frauds have
median `Amount` **$2.99** against **$33.59** for caught frauds; 13 of 22 misses
are under $5. Across the full dataset, **36.2% of frauds are ≤ $1.00** versus
10.6% of legitimate transactions — a distinct card-testing mode where the
attacker probes with a micro-authorisation. **Caveat: 22 missed frauds is a very
small sample**, so treat the direction as suggestive and the effect size as
unestimated. The full-dataset $1 statistic does not depend on that sample.

---

## 8. Business recommendations

Each recommendation cites the SHAP evidence it rests on, and the cost of acting
on it.

### R1 — Step-up verification for any purchase within 1 hour of signup
**Evidence.** SHAP for `time_since_signup` ≤ 1 h is **+4.04 log-odds**, ~17× the
magnitude of any other bucket, and the effect collapses to +1.34 by hour six
(§3.3). The true positive in §4.1 purchased **one second** after signup. As a
standalone rule: **99.52% precision, 53.7% recall, 37 false positives in
151,112 transactions.**

**Action.** Require 3-D Secure / OTP before authorising any first purchase in the
first hour of an account's life. Within the **first minute** (7,600 transactions,
**100.0% fraud**), decline outright.

**Cost.** ~37 legitimate customers per 151k transactions see one extra
verification step — **0.02%** of traffic. This is the highest-value, lowest-cost
intervention available and needs no model in production to enforce.

### R2 — Cap distinct accounts per device, and block on 24-hour velocity ≥ 2
**Evidence.** `device_transaction_count` is the **top SHAP driver** (58.6% of
total). Its effect flips sign at exactly 2: fraud rate **3.07% → 23.40%** (7.6×),
reaching 86–90% at 5+ (§3.3). `device_velocity_24h` is the sharpest boundary in
the data: **5.13% fraud at 1 purchase, 94.89% at 2, ~100% at 3+**. The §4.4
contrast shows this feature alone flipping the verdict at constant account age.

**Action.** Two tiers. (a) **Velocity:** a second purchase from the same device
within 24 hours triggers step-up verification — 99.29% precision, 48.4% recall,
49 false positives. (b) **Device sharing:** at the 5th distinct account on one
device, freeze the device pending manual review — 90.87% precision.

**Cost.** Tier (a) is nearly free (49 false positives / 151k). Tier (b) costs 766
false positives; if that is too aggressive, raise the trigger to 6+ accounts,
where the fraud rate is 86–90%.

### R3 — Deploy the two rules as a pre-filter, and the model as the scoring layer
**Evidence.** §7.3: the union `tsu ≤ 1h OR velocity ≥ 2` achieves **98.88%
precision at 53.8% recall**, statistically indistinguishable from the tuned
XGBoost model (99.93% / 52.7%).

**Action.** Run the rules first — they are deterministic, instant, need no
feature pipeline, and are explainable to a regulator in one sentence. Route
everything they *don't* flag to the model, which contributes calibrated scores
for triage ranking and catches combinations the rules miss. Keep the model in the
loop for drift detection: **when the rules and the model start disagreeing, the
fraud pattern has changed** — that divergence is a free early-warning signal.

**Cost.** One extra service in the path. The upside is that ~5% of traffic
resolves before the model is invoked.

### R4 — Stop scoring `device_user_count` and `user_transaction_count`
**Evidence.** §7.1: `device_user_count` is *identical* to
`device_transaction_count` on every row (r = 1.000000);
`user_transaction_count` is constant at 1. The duplicate's gain rank (#2) is an
artefact — SHAP puts it at #4 with 4.9% share, all of it stolen from its twin.

**Action.** Drop both from the feature set and re-fit. Expect **no** performance
change; the gain ranking becomes honest, SHAP explanations become non-redundant,
and analyst review time stops being spent on a phantom second signal.

**Also fix the upstream cause.** `user_transaction_count` is constant *only
because this extract holds one row per user*. On a production stream with repeat
customers it would carry real signal. The right fix is to compute these features
against the **full account history**, not the extract — at which point
`device_user_count` and `device_transaction_count` genuinely diverge and the
feature becomes useful.

### R5 — Do not chase the residual 28% with modelling; buy new signal instead
**Evidence.** §7.4: 4,008 frauds sit in a region where fraudulent and legitimate
transactions have matching medians on every available feature. Every Task-2
model plateaued at the same ~71% recall as a one-line rule.

**Action.** Redirect effort from tuning to data acquisition. The failure mode in
§4.3 — fraud on a **fresh, unshared device outside the 1-hour window** — points at
what is missing: payment-instrument history (card BIN reuse, prior chargebacks),
billing-vs-shipping address mismatch, email/phone age and reputation, and
device-fingerprint stability across sessions. Each attacks the blind spot
directly; none is derivable from the current columns.

### R6 — Add a velocity rule for micro-authorisations on the card portfolio
**Evidence.** §7.5: missed frauds have median `Amount` $2.99 vs $33.59 for caught
frauds, and **36.2% of all frauds are ≤ $1.00** against 10.6% of legitimate
transactions — the card-testing signature, confirmed by the §4.5 false negative
at $1.18 with no PCA-space trace.

**Action.** Amount alone cannot carry this rule (10.6% of legitimate traffic is
also ≤ $1). Use *velocity of small authorisations per card*: three or more
authorisations under $1 on one card within an hour should trigger a block
regardless of model score.

**Caveat, stated plainly.** The 22-miss sample supports the *direction*, not a
threshold. Validate "3 in 1 hour" against production data before enforcing it —
the `creditcard` extract has no card identifier, so it cannot be validated here.

---

## 9. Reproducibility

```bash
pytest                                          # 90 tests
python scripts/explain_models.py                # all Task 3 figures & tables
python scripts/explain_models.py --dataset fraud
jupyter nbconvert --to notebook --execute --inplace notebooks/shap-explainability.ipynb
```

| Artifact | Path |
|---|---|
| Built-in importance (gain + weight) | `reports/task3_builtin_importance_{fraud,creditcard}.csv` |
| SHAP importance (abs + signed) | `reports/task3_shap_importance_{fraud,creditcard}.csv` |
| Gain vs SHAP comparison | `reports/task3_importance_comparison_{fraud,creditcard}.csv` |
| Effect-shape tables | `reports/task3_effect_shape_{fraud,creditcard}.csv` |
| Selected cases | `reports/task3_cases_{fraud,creditcard}.csv` |
| Local contributions | `reports/task3_local_contributions_{fraud,creditcard}.csv` |
| Importance plots | `reports/figures/{fraud,creditcard}_builtin_importance.png` |
| SHAP summary / bar | `reports/figures/{fraud,creditcard}_shap_{summary,bar}.png` |
| Force plots | `reports/figures/{fraud,creditcard}_force_{true_positive,false_positive,false_negative,true_negative}.png` |
| Waterfall plots | `reports/figures/{fraud,creditcard}_waterfall_*.png` |
| Dependence plots | `reports/figures/{fraud,creditcard}_dependence_*.png` |
| Narrative | `notebooks/shap-explainability.ipynb` |
| Code | `src/explainability.py`, `scripts/explain_models.py` |

SHAP additivity is asserted in the test suite
(`test_shap_values_sum_to_model_margin`): explanations are verified to reconstruct
the model's raw margin, so no figure in this report rests on an unchecked
attribution.
