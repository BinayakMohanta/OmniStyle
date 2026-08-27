"""
src/train_model.py
====================

Two modeling tasks, run sequentially:

1. CUSTOMER SEGMENTATION (unsupervised)
   - Scale RFM features
   - Choose k via elbow + silhouette analysis
   - Fit K-Means
   - Derive human-readable segment labels from cluster characteristics
     (not from arbitrary cluster IDs)
   - Save customer_segments.csv + visualizations

2. CHURN PREDICTION (supervised)
   - Baseline: Logistic Regression
   - Advanced: Random Forest
   - Stratified train/test split, cross-validation, class-imbalance handling
   - Evaluate with precision/recall/F1/ROC-AUC/confusion matrix
   - Select best model by ROC-AUC (business-relevant: ranking at-risk customers)
   - SHAP explainability for the tree-based model
   - Save best_churn_model.joblib, metrics, predictions, and plots

Also generates dashboard/sample_data/ CSVs for Power BI and
reports/business_recommendations.md summarizing actionable insights.

Run as:
    python -m src.train_model
"""

from __future__ import annotations

import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    DASHBOARD_SAMPLE_DATA_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    get_logger,
)

# Segmentation logic now lives in src/customer_segmentation.py (single source
# of truth). Imported here so existing callers of `src.train_model` keep
# working unchanged (e.g. `from src.train_model import label_segments` in
# notebooks/03_segmentation.ipynb).
from src.customer_segmentation import (
    SEGMENT_FEATURES,
    choose_k,
    label_segments,
    run_segmentation,
)

logger = get_logger(__name__)

np.random.seed(RANDOM_SEED)


# =============================================================================
# 1. CUSTOMER SEGMENTATION
# =============================================================================
# NOTE: choose_k, label_segments, and run_segmentation are imported above
# from src.customer_segmentation. run_segmentation() already saves
# data/processed/customer_segments.csv and the reports/*.png diagnostics,
# exactly as this module used to do inline, so the rest of the training
# pipeline below (churn model, SHAP, dashboard export, recommendations)
# requires no changes.


# =============================================================================
# 2. CHURN PREDICTION
# =============================================================================
CHURN_FEATURE_COLUMNS = [
    "age", "customer_tenure_days", "recency_days", "frequency", "monetary",
    "avg_order_value", "total_items_purchased", "unique_products_purchased",
    "unique_categories_purchased", "average_discount", "purchase_trend_ratio",
    "loyalty_points", "loyalty_tier_encoded",
]


def _load_churn_dataset() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "churn_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python -m src.feature_engineering' first.")
    return pd.read_csv(path)


def train_churn_models(df: pd.DataFrame) -> dict:
    logger.info("--- Churn Prediction ---")
    X = df[CHURN_FEATURE_COLUMNS].fillna(0)
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    logger.info("Train size: %d | Test size: %d | Churn rate (train): %.1f%%",
                len(X_train), len(X_test), y_train.mean() * 100)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_weight = "balanced"  # handle class imbalance

    # --- Baseline: Logistic Regression ---
    logreg = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=RANDOM_SEED)
    logreg.fit(X_train_scaled, y_train)
    logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_pred = logreg.predict(X_test_scaled)

    # --- Advanced: Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight=class_weight, random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(X_train, y_train)  # tree models don't need scaling
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)

    # --- Cross-validation (ROC-AUC) for robustness check ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rf_cv_scores = []
    for train_idx, val_idx in cv.split(X, y):
        model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight=class_weight, random_state=RANDOM_SEED, n_jobs=-1,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        rf_cv_scores.append(roc_auc_score(y.iloc[val_idx], proba))
    logger.info("Random Forest 5-fold CV ROC-AUC: mean=%.3f, std=%.3f", np.mean(rf_cv_scores), np.std(rf_cv_scores))

    def _metrics(name, y_true, y_pred, y_proba):
        return {
            "model": name,
            "precision": round(precision_score(y_true, y_pred), 3),
            "recall": round(recall_score(y_true, y_pred), 3),
            "f1_score": round(f1_score(y_true, y_pred), 3),
            "roc_auc": round(roc_auc_score(y_true, y_proba), 3),
        }

    logreg_metrics = _metrics("Logistic Regression", y_test, logreg_pred, logreg_proba)
    rf_metrics = _metrics("Random Forest", y_test, rf_pred, rf_proba)

    logger.info("Logistic Regression metrics: %s", logreg_metrics)
    logger.info("Random Forest metrics: %s", rf_metrics)
    logger.info("Full classification report (Random Forest):\n%s",
                classification_report(y_test, rf_pred, target_names=["Not Churned", "Churned"]))

    # Select best model primarily on ROC-AUC (ranking quality for targeting
    # retention campaigns matters more than raw accuracy for this business
    # use case), with F1 as a tie-breaker.
    if rf_metrics["roc_auc"] >= logreg_metrics["roc_auc"]:
        best_name, best_model, best_proba, best_pred = "Random Forest", rf, rf_proba, rf_pred
    else:
        best_name, best_model, best_proba, best_pred = "Logistic Regression", logreg, logreg_proba, logreg_pred

    logger.info("Selected best model: %s", best_name)

    # Save confusion matrix plot for the best model
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, best_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Not Churned", "Churned"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    cm_path = REPORTS_DIR / "churn_confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    plt.close(fig)

    # Save model, scaler (needed only for logistic regression), and metrics
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "best_churn_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")

    metrics_summary = {
        "best_model": best_name,
        "logistic_regression": logreg_metrics,
        "random_forest": rf_metrics,
        "random_forest_cv_roc_auc_mean": round(float(np.mean(rf_cv_scores)), 3),
        "random_forest_cv_roc_auc_std": round(float(np.std(rf_cv_scores)), 3),
        "churn_feature_columns": CHURN_FEATURE_COLUMNS,
    }
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    # Also mirror the metrics into reports/ so tooling that expects metrics
    # alongside the other human-readable reports (SHAP plots, business
    # recommendations, etc.) can find them without reaching into models/.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("Saved best model (%s) and metrics to %s and %s", best_name, MODELS_DIR, REPORTS_DIR)

    # Save predictions for downstream use (e.g., Power BI / recommendations)
    predictions_df = df.loc[X_test.index, ["customer_id"]].copy()
    predictions_df["actual_churned"] = y_test.values
    predictions_df["predicted_churned"] = best_pred
    predictions_df["churn_probability"] = np.round(best_proba, 4)
    predictions_df.to_csv(DATA_PROCESSED_DIR / "churn_predictions.csv", index=False)

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "is_tree_based": best_name == "Random Forest",
        "X_train": X_train, "X_test": X_test, "y_test": y_test,
        "metrics_summary": metrics_summary,
        "predictions_df": predictions_df,
    }


def run_shap_explainability(model_bundle: dict) -> None:
    if not model_bundle["is_tree_based"]:
        logger.info("Best model is not tree-based; skipping SHAP TreeExplainer (would require KernelExplainer, slower).")
        return

    import shap

    logger.info("--- SHAP Explainability ---")
    model = model_bundle["best_model"]
    X_test = model_bundle["X_test"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP's output shape for binary classifiers varies by version:
    #   - list of two arrays [class0, class1], each (n_samples, n_features)
    #   - a single 3D array (n_samples, n_features, n_classes)
    #   - a single 2D array (n_samples, n_features) already for the positive class
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values_pos = shap_values[:, :, 1]
    else:
        shap_values_pos = shap_values

    # Global feature importance (summary plot)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values_pos, X_test, show=False)
    plt.tight_layout()
    summary_path = REPORTS_DIR / "shap_summary_plot.png"
    plt.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot to %s", summary_path)

    # Global mean |SHAP value| bar chart
    mean_abs_shap = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": np.abs(shap_values_pos).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    mean_abs_shap.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(mean_abs_shap["feature"][:10][::-1], mean_abs_shap["mean_abs_shap"][:10][::-1], color="indianred")
    ax.set_title("Top 10 Churn Drivers (Mean |SHAP value|)")
    ax.set_xlabel("Mean |SHAP value| (impact on model output)")
    plt.tight_layout()
    bar_path = REPORTS_DIR / "shap_feature_importance_bar.png"
    plt.savefig(bar_path, dpi=120)
    plt.close(fig)
    logger.info("Saved SHAP feature importance bar chart to %s", bar_path)

    # Local explanation for a single at-risk customer (highest predicted churn probability)
    preds = model_bundle["predictions_df"]
    top_risk_idx_label = preds.sort_values("churn_probability", ascending=False).index[0]
    row_position = X_test.index.get_loc(top_risk_idx_label)

    expected_value = explainer.expected_value
    expected_value_arr = np.atleast_1d(expected_value)
    expected_value = expected_value_arr[-1] if expected_value_arr.size > 1 else expected_value_arr[0]

    try:
        explanation = shap.Explanation(
            values=shap_values_pos[row_position],
            base_values=expected_value,
            data=X_test.iloc[row_position].values,
            feature_names=list(X_test.columns),
        )
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(explanation, show=False, max_display=10)
        plt.tight_layout()
        local_path = REPORTS_DIR / "shap_local_explanation_example.png"
        plt.savefig(local_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved local SHAP explanation example to %s", local_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not render SHAP local waterfall plot (%s). Global plots are still available.", exc)


# =============================================================================
# 3. DASHBOARD (POWER BI) EXPORTS
# =============================================================================
def export_dashboard_data(segments_df: pd.DataFrame, churn_bundle: dict) -> None:
    logger.info("--- Exporting Power BI-ready datasets ---")

    customers = pd.read_csv(DATA_PROCESSED_DIR / "customers_clean.csv")
    products = pd.read_csv(DATA_PROCESSED_DIR / "products_clean.csv")
    orders = pd.read_csv(DATA_PROCESSED_DIR / "orders_clean.csv")
    order_items = pd.read_csv(DATA_PROCESSED_DIR / "order_items_clean.csv")
    features = pd.read_csv(DATA_PROCESSED_DIR / "customer_features.csv")

    sales_detail = order_items.merge(
        orders[["order_id", "customer_id", "order_date", "channel", "store_region", "payment_method"]],
        on="order_id",
    ).merge(products[["product_id", "product_name", "category", "subcategory", "brand"]], on="product_id")

    sales_detail.to_csv(DASHBOARD_SAMPLE_DATA_DIR / "sales_detail.csv", index=False)
    customers.to_csv(DASHBOARD_SAMPLE_DATA_DIR / "customers.csv", index=False)
    products.to_csv(DASHBOARD_SAMPLE_DATA_DIR / "products.csv", index=False)
    orders.to_csv(DASHBOARD_SAMPLE_DATA_DIR / "orders.csv", index=False)

    customer_360 = (
        customers.merge(features, on="customer_id", how="left", suffixes=("", "_feat"))
        .merge(segments_df[["customer_id", "segment"]], on="customer_id", how="left")
        .merge(churn_bundle["predictions_df"][["customer_id", "churn_probability"]], on="customer_id", how="left")
    )
    customer_360.to_csv(DASHBOARD_SAMPLE_DATA_DIR / "customer_360.csv", index=False)

    logger.info("Power BI-ready CSVs written to %s", DASHBOARD_SAMPLE_DATA_DIR)


# =============================================================================
# 4. BUSINESS RECOMMENDATIONS
# =============================================================================
def generate_business_recommendations(segments_df: pd.DataFrame, churn_bundle: dict) -> None:
    logger.info("--- Generating business_recommendations.md ---")

    segment_sizes = segments_df["segment"].value_counts()
    segment_pct = (segment_sizes / segment_sizes.sum() * 100).round(1)

    preds = churn_bundle["predictions_df"]
    high_risk_count = (preds["churn_probability"] >= 0.7).sum()
    avg_churn_prob = preds["churn_probability"].mean()

    shap_importance_path = REPORTS_DIR / "shap_feature_importance.csv"
    top_drivers_text = ""
    if shap_importance_path.exists():
        top_drivers = pd.read_csv(shap_importance_path).head(5)
        top_drivers_text = "\n".join(
            f"{i+1}. **{row['feature']}** (mean |SHAP| = {row['mean_abs_shap']:.4f})"
            for i, row in top_drivers.iterrows()
        )

    metrics = churn_bundle["metrics_summary"]
    best_model_name = metrics["best_model"]
    best_metrics = metrics["random_forest"] if best_model_name == "Random Forest" else metrics["logistic_regression"]

    content = f"""# OmniStyle Business Recommendations

*Generated automatically from the trained segmentation and churn models. All
figures below come directly from `data/processed/customer_segments.csv`,
`data/processed/churn_predictions.csv`, and `reports/shap_feature_importance.csv`.*

## 1. Customer Segment Overview

| Segment | Customers | % of Base |
|---|---|---|
{chr(10).join(f"| {seg} | {segment_sizes[seg]} | {segment_pct[seg]}% |" for seg in segment_sizes.index)}

## 2. Churn Model Performance

The selected model is **{best_model_name}**, chosen primarily on ROC-AUC
(ranking quality is more useful than raw accuracy for prioritizing retention
outreach). On the held-out test set:

- Precision: {best_metrics['precision']}
- Recall: {best_metrics['recall']}
- F1-score: {best_metrics['f1_score']}
- ROC-AUC: {best_metrics['roc_auc']}

Across the scored test population, **{high_risk_count} customers** have a
predicted churn probability of 70% or higher, and the average predicted
churn probability is **{avg_churn_prob:.1%}**.

## 3. Top Churn Drivers (SHAP Global Importance)

{top_drivers_text if top_drivers_text else "Run `python -m src.train_model` to populate SHAP-based drivers."}

These are the features that most strongly push the model's churn prediction
up or down, in order of overall impact.

## 4. Actionable Recommendations by Segment

### Champions & Loyal Customers
High-value, high-risk customers who show increasing inactivity are the most
expensive to lose and the most likely to respond to attention. Any customer
in these segments whose `churn_probability` (see
`data/processed/churn_predictions.csv`) exceeds 0.5 should be prioritized for
a **personal retention outreach** (loyalty bonus, early access, or a
dedicated support check-in) rather than a generic discount blast.

### Potential Loyalists & New Customers
These customers are recent but not yet frequent. A **structured onboarding
journey** — a second-purchase incentive, category-based product
recommendations, and loyalty-program enrollment — is more cost-effective
here than deep discounting, since their price sensitivity is not yet proven.

### At Risk
Customers with high historical frequency/monetary value but rising recency
(days since last purchase) should receive **win-back campaigns** timed
around their historical purchase cadence, referencing categories they have
bought before.

### Hibernating
Low frequency, low monetary, long recency. Because the SHAP analysis shows
`recency_days` and `frequency` as dominant churn drivers, these customers are
already highly likely to churn or have effectively churned. **Do not spend
premium retention budget here** — a low-cost, automated re-engagement email
is appropriate, but expensive incentives should be reserved for
higher-value, higher-probability-of-return segments.

## 5. Product & Category Actions

Cross-reference `sql/business_analysis.sql` query #5 (bottom-performing
products) with the category performance query (#6). Categories or SKUs that
combine **low revenue share** with **high discount dependency** (see
business_analysis.sql query #17) should be reviewed for markdown, bundling,
or discontinuation, since they are consuming margin without driving repeat
purchases.

## 6. Suggested Next Steps for the Business

1. Build a weekly export of `churn_probability >= 0.6` customers from
   `data/processed/churn_predictions.csv` and route it to the retention team.
2. Track month-over-month movement of customers between segments in
   `customer_segments.csv` as a leading indicator of overall portfolio health.
3. A/B test retention offers on the "At Risk" segment specifically, since it
   is the segment where incremental spend is most likely to be recovered
   before customers fully churn.
"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "business_recommendations.md"
    out_path.write_text(content)
    logger.info("Saved %s", out_path)


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    logger.info("=" * 70)
    logger.info("OmniStyle Customer Intelligence Platform - Model Training")
    logger.info("=" * 70)

    segments_df = run_segmentation()
    churn_bundle = train_churn_models(_load_churn_dataset())

    try:
        run_shap_explainability(churn_bundle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP explainability step failed (%s). Continuing without it.", exc)

    export_dashboard_data(segments_df, churn_bundle)
    generate_business_recommendations(segments_df, churn_bundle)

    logger.info("Model training complete. See models/, reports/, and dashboard/sample_data/ for outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
