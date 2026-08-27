# Methodology

This document describes exactly what the code does — no methodology is
documented here that the implementation doesn't actually follow.

## 1. Synthetic Data Generation (`src/data_ingestion.py`)

Data is generated deterministically (seeded by `RANDOM_SEED`, default 42)
rather than fully independently at random, so that downstream analytics and
ML are meaningful:

- Each customer is assigned one of six **behavioral archetypes**
  (`highly_active`, `loyal`, `new_customer`, `at_risk`, `inactive`,
  `churned`), each with its own weight, monthly order-rate range, recency
  bias range, and discount-affinity range (`ARCHETYPES` in
  `src/data_ingestion.py`).
- Archetype drives correlated behavior across signup recency, loyalty tier,
  order frequency, order recency, and discount sensitivity — e.g. `loyal`
  and `highly_active` customers skew toward longer tenure and higher
  loyalty tiers; `churned`/`inactive` customers skew toward very old last-
  purchase dates.
- Orders are seasonally weighted (`SEASONAL_MULTIPLIER`) to bias volume
  toward back-to-school (Aug/Sep) and holiday (Nov/Dec) months.
- Region/category affinity weights (`REGION_CATEGORY_AFFINITY`) bias which
  products a customer in a given region is more likely to buy.
- Target volumes default to ~5,000 customers, ~200 products, and a
  ~35,000-order target (the realized order count is somewhat lower after
  Poisson sampling per customer — typically ~25,000–30,000 — which is
  within the intended 25,000–50,000 range).

## 2. Cleaning (`src/data_cleaning.py`)

Each table is validated independently, then cross-validated against the
others:
- Duplicate primary keys are dropped (first occurrence kept).
- Invalid numeric ranges (negative prices/costs/quantities, out-of-range
  ages, discounts outside [0, 0.9]) are corrected — either clipped or
  replaced with a sensible fallback (e.g. median), and the exact
  count/action is logged to `data_quality_report.csv`.
- `orders.total_amount` is reconciled against the sum of that order's
  `order_items.line_total`. Mismatches beyond a $1 tolerance are corrected
  using the line-item sum (line items are the source of truth).
- Orphaned foreign keys (an order item referencing a since-removed order or
  product; an order referencing a since-removed customer) are dropped.
- Orders that end up with zero valid line items after cleaning are dropped
  entirely.

## 3. Feature Engineering (`src/feature_engineering.py`)

### RFM (Recency, Frequency, Monetary)

Computed as of a fixed **cutoff date** (`CUTOFF_DATE`, currently 90 days
before the project's reference date):
- **Recency** = days between the cutoff and the customer's most recent
  order on/before the cutoff (or, if they have no orders yet, days since
  signup).
- **Frequency** = count of orders on/before the cutoff.
- **Monetary** = sum of `total_amount` for orders on/before the cutoff.
- RFM values are additionally scored into quintiles (1–5, 5 = best) via
  `add_rfm_scores`, producing `R_score`, `F_score`, `M_score`, and a
  concatenated `RFM_score` string.

### Broader customer features

`build_customer_features` extends RFM with tenure, item/category diversity,
average discount, and a **purchase trend ratio**: total spend in the 90
days immediately before the cutoff, divided by spend in the 90 days before
that (a simple momentum signal; capped conceptually via a default of 1.0
when there's no prior-period spend to compare against, or 2.0 when there is
recent spend but literally zero prior-period spend).

### Churn label — leakage prevention

This is the most safety-critical piece of the pipeline, so the exact
mechanism is repeated here in full:

- A fixed **cutoff date** splits history into a **feature window**
  (everything on/before the cutoff) and a 90-day **observation window**
  (strictly after the cutoff, up to `cutoff + 90 days`).
- **All features are computed only from the feature window.** The
  observation window's orders are never used to build recency, frequency,
  monetary, or any other feature — only to compute the label.
- **The label is computed only from the observation window.** A customer is
  labeled `churned = 1` if they have **zero** orders in the observation
  window; `churned = 0` if they have at least one.
- Only customers who **signed up on or before the cutoff date** are
  eligible for labeling — this ensures every labeled customer genuinely had
  the opportunity to place an order in the observation window (a customer
  who signed up after the cutoff hasn't had a fair chance to be "active"
  yet, so including them would bias the label distribution).

Because the feature window and observation window are strictly
non-overlapping in time, and the model is only ever trained on features
from the feature window, there is no leakage of future information into
the churn model.

## 4. Customer Segmentation (`src/customer_segmentation.py`)

1. Features: `recency_days`, `frequency`, `monetary` from
   `rfm_features.csv`, standardized with `sklearn.preprocessing.StandardScaler`.
2. **k selection**: K-Means is fit for each `k` in `range(4, 8)`, and both
   inertia (for an elbow-style visual) and silhouette score (computed on a
   random sample of up to 2,000 points, for speed) are recorded. The `k`
   with the highest silhouette score in that range is selected.
   - The search intentionally starts at `k=4` rather than `k=2`: on this
     dataset, unconstrained silhouette maximization tends to collapse to a
     trivial `k=2` "active vs. inactive" split, which is statistically
     clean but not business-actionable. Constraining to `[4, 7]` produces
     segmentation that maps onto standard, actionable RFM personas.
3. **Cluster labeling**: clusters are profiled by their mean
   recency/frequency/monetary, then each cluster's *rank* on each dimension
   **relative to the other clusters found in that run** (not a fixed
   population-wide threshold, and not the raw cluster ID) determines its
   business label: `Champions`, `Loyal Customers`, `Potential Loyalists`,
   `New Customers`, `At Risk`, or `Hibernating` (see
   `label_segments` for the exact rule set).

## 5. Churn Prediction (`src/train_model.py`)

- **Features**: the 13 columns in `CHURN_FEATURE_COLUMNS` (age, tenure,
  RFM-derived features, item/category diversity, discount behavior, trend
  ratio, loyalty fields).
- **Split**: stratified 80/20 train/test split (`train_test_split`,
  `stratify=y`, `random_state=RANDOM_SEED`) so the churn rate is preserved
  in both sets.
- **Baseline model**: Logistic Regression (`class_weight="balanced"`,
  `max_iter=1000`) on standardized features.
- **Advanced model**: Random Forest (`n_estimators=300`, `max_depth=8`,
  `min_samples_leaf=5`, `class_weight="balanced"`) on raw (unscaled)
  features, since tree-based models don't require scaling.
- **Class imbalance**: handled via `class_weight="balanced"` on both models
  rather than resampling, keeping the training set representative of the
  true population.
- **Cross-validation**: a separate 5-fold `StratifiedKFold` cross-validation
  of the Random Forest reports mean/std ROC-AUC as a robustness check
  beyond the single train/test split.
- **Evaluation**: precision, recall, F1-score, ROC-AUC, and a confusion
  matrix plot, computed on the held-out test set for both models.
- **Model selection**: the model with the **higher ROC-AUC** is selected as
  `best_model`, because for this use case (ranking customers by churn risk
  to prioritize retention outreach) ranking quality matters more than raw
  classification accuracy. All metrics for both models are still saved to
  `models/model_metrics.json` for transparency.
- **Persistence**: the selected model → `models/best_churn_model.joblib`;
  the fitted `StandardScaler` → `models/feature_scaler.joblib` (used if the
  selected model is Logistic Regression); metrics → `models/model_metrics.json`;
  per-customer test-set predictions → `data/processed/churn_predictions.csv`.

## 6. Explainability (`src/train_model.py::run_shap_explainability`)

Applied only when the selected model is tree-based (Random Forest in
practice, given the ROC-AUC comparison above tends to favor it):

- `shap.TreeExplainer` computes SHAP values for the test set.
- A **global summary plot** (`shap_summary_plot.png`) shows the
  distribution of each feature's impact across all test customers.
- A **ranked bar chart** (`shap_feature_importance_bar.png`) and CSV
  (`shap_feature_importance.csv`) summarize mean absolute SHAP value per
  feature — i.e., overall importance.
- A **local waterfall explanation** (`shap_local_explanation_example.png`)
  is generated for the single test customer with the highest predicted
  churn probability, showing exactly which features pushed their score up
  or down.

## 7. Recommendations & Power BI export

- `src/generate_recommendations.py` recomputes segment summaries, churn
  risk buckets, category/regional performance, and bottom products directly
  from the processed CSVs and model outputs each time it runs, and renders
  `reports/business_recommendations.md` purely from those computed values —
  no hardcoded business figures.
- `src/export_dashboard_data.py` reshapes the same underlying cleaned data
  and model outputs into business-friendly, denormalized CSVs suitable for
  direct import into Power BI, without altering the underlying values.

## 8. Reproducibility

`RANDOM_SEED` (default 42) seeds Python's `random` module, NumPy, and every
scikit-learn estimator that accepts a `random_state`. Given the same seed
and the same synthetic-data volume parameters (`NUM_CUSTOMERS`,
`NUM_PRODUCTS`, `NUM_ORDERS`), re-running the full pipeline via
`python -m src.run_pipeline` reproduces the same data, features, segments,
and model metrics.
