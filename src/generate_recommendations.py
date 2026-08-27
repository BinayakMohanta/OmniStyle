"""
src/generate_recommendations.py
==================================

Standalone business recommendations generator.

`src/train_model.py` already writes a first version of
`reports/business_recommendations.md` immediately after training (short
segment/churn summary). This module is the canonical, more complete
generator: it reads every real, already-generated artifact from disk
(segments, churn predictions, SHAP importances, model metrics, cleaned
sales data) and produces a fuller report with the sections requested for
the completed project (Executive Summary, Key Business Findings, Customer
Retention Recommendations, Segment-Specific Actions, Churn Risk Actions,
Product/Category Opportunities, Regional Opportunities, Recommended Next
Steps). Running this script overwrites `reports/business_recommendations.md`
with this fuller version — that is expected and intentional, since it is
meant to be the final report-generation step in the pipeline.

Every number in the generated report is computed directly from the project's
own CSV/JSON outputs. Nothing is hardcoded or invented.

Run as:
    python -m src.generate_recommendations
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from src.config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, get_logger

logger = get_logger(__name__)


def _require(path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}\n{hint}")


def load_inputs() -> dict:
    segments_path = DATA_PROCESSED_DIR / "customer_segments.csv"
    churn_path = DATA_PROCESSED_DIR / "churn_predictions.csv"
    features_path = DATA_PROCESSED_DIR / "customer_features.csv"
    customers_path = DATA_PROCESSED_DIR / "customers_clean.csv"
    products_path = DATA_PROCESSED_DIR / "products_clean.csv"
    orders_path = DATA_PROCESSED_DIR / "orders_clean.csv"
    order_items_path = DATA_PROCESSED_DIR / "order_items_clean.csv"

    _require(segments_path, "Run 'python -m src.customer_segmentation' first.")
    _require(churn_path, "Run 'python -m src.train_model' first.")
    _require(features_path, "Run 'python -m src.feature_engineering' first.")

    data = {
        "segments": pd.read_csv(segments_path),
        "churn_predictions": pd.read_csv(churn_path),
        "features": pd.read_csv(features_path),
        "customers": pd.read_csv(customers_path),
        "products": pd.read_csv(products_path),
        "orders": pd.read_csv(orders_path),
        "order_items": pd.read_csv(order_items_path),
    }

    # model_metrics.json is written to both models/ and reports/ by
    # src/train_model.py; prefer reports/ (co-located with the other
    # human-readable outputs) but fall back to models/ for compatibility.
    metrics_path = REPORTS_DIR / "model_metrics.json"
    if not metrics_path.exists():
        metrics_path = MODELS_DIR / "model_metrics.json"
    data["model_metrics"] = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    shap_path = REPORTS_DIR / "shap_feature_importance.csv"
    data["shap_importance"] = pd.read_csv(shap_path) if shap_path.exists() else None

    return data


def compute_segment_summary(data: dict) -> pd.DataFrame:
    segments = data["segments"]
    summary = (
        segments.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .round(1)
        .sort_values("avg_monetary", ascending=False)
    )
    summary["pct_of_base"] = (summary["customers"] / summary["customers"].sum() * 100).round(1)
    return summary


def compute_churn_summary(data: dict) -> dict:
    preds = data["churn_predictions"]
    return {
        "scored_customers": len(preds),
        "avg_churn_probability": preds["churn_probability"].mean(),
        "high_risk_count": int((preds["churn_probability"] >= 0.7).sum()),
        "medium_risk_count": int(
            ((preds["churn_probability"] >= 0.4) & (preds["churn_probability"] < 0.7)).sum()
        ),
        "low_risk_count": int((preds["churn_probability"] < 0.4).sum()),
        "actual_churn_rate_in_test_set": preds["actual_churned"].mean(),
    }


def compute_category_performance(data: dict) -> pd.DataFrame:
    sales = data["order_items"].merge(data["products"][["product_id", "category"]], on="product_id", how="left")
    perf = sales.groupby("category").agg(
        revenue=("line_total", "sum"),
        units_sold=("quantity", "sum"),
        avg_discount=("discount", "mean"),
    ).round(2).sort_values("revenue", ascending=False)
    perf["pct_of_revenue"] = (perf["revenue"] / perf["revenue"].sum() * 100).round(1)
    return perf


def compute_regional_performance(data: dict) -> pd.DataFrame:
    sales = data["order_items"].merge(
        data["orders"][["order_id", "store_region", "customer_id"]], on="order_id", how="left"
    )
    perf = sales.groupby("store_region").agg(
        revenue=("line_total", "sum"),
        orders=("order_id", "nunique"),
        customers=("customer_id", "nunique"),
    ).round(2).sort_values("revenue", ascending=False)
    perf["avg_order_value"] = (perf["revenue"] / perf["orders"]).round(2)
    return perf


def compute_bottom_products(data: dict, n: int = 5) -> pd.DataFrame:
    sales = data["order_items"].merge(data["products"], on="product_id", how="left")
    perf = sales.groupby(["product_id", "product_name", "category"]).agg(
        revenue=("line_total", "sum"),
        avg_discount=("discount", "mean"),
    ).reset_index()
    return perf.sort_values("revenue", ascending=True).head(n)


def render_markdown(data: dict) -> str:
    segment_summary = compute_segment_summary(data)
    churn_summary = compute_churn_summary(data)
    category_perf = compute_category_performance(data)
    regional_perf = compute_regional_performance(data)
    bottom_products = compute_bottom_products(data)

    total_revenue = data["order_items"]["line_total"].sum()
    total_customers = len(data["customers"])
    total_orders = data["orders"]["order_id"].nunique()

    model_metrics = data["model_metrics"]
    if model_metrics:
        best_model_name = model_metrics["best_model"]
        best_key = "random_forest" if best_model_name == "Random Forest" else "logistic_regression"
        best_metrics = model_metrics[best_key]
        model_section = (
            f"The production churn model is **{best_model_name}**, selected primarily on "
            f"ROC-AUC ({best_metrics['roc_auc']}) since ranking customers by churn risk matters "
            f"more for retention targeting than raw classification accuracy. On the held-out "
            f"test set it achieves precision {best_metrics['precision']}, recall "
            f"{best_metrics['recall']}, and F1-score {best_metrics['f1_score']}."
        )
    else:
        model_section = "Model metrics were not found; run `python -m src.train_model` first."

    shap_section = "SHAP feature importance was not found; run `python -m src.train_model` first."
    if data["shap_importance"] is not None:
        top5 = data["shap_importance"].head(5)
        shap_section = "\n".join(
            f"{i+1}. **{row['feature']}** (mean |SHAP| = {row['mean_abs_shap']:.4f})"
            for i, row in top5.reset_index(drop=True).iterrows()
        )

    top_category = category_perf.index[0]
    top_category_pct = category_perf.iloc[0]["pct_of_revenue"]
    worst_category = category_perf.index[-1]
    worst_category_pct = category_perf.iloc[-1]["pct_of_revenue"]

    top_region = regional_perf.index[0]
    worst_region = regional_perf.index[-1]

    at_risk_row = segment_summary.loc["At Risk"] if "At Risk" in segment_summary.index else None
    hibernating_row = segment_summary.loc["Hibernating"] if "Hibernating" in segment_summary.index else None

    content = f"""# OmniStyle Business Recommendations

*Auto-generated by `src/generate_recommendations.py` from the project's own
processed data, segmentation output, churn predictions, and model metrics.
Every figure below is computed directly from those files — nothing here is
placeholder text.*

## Executive Summary

OmniStyle's data covers **{total_customers:,} customers**, **{total_orders:,} orders**,
and **${total_revenue:,.2f}** in recorded revenue. Customer behavior splits
into {len(segment_summary)} data-driven segments (see below), and a trained
churn model scores **{churn_summary['scored_customers']:,} customers** on a
held-out test set, flagging **{churn_summary['high_risk_count']:,}** as
high-risk (churn probability ≥ 70%). {model_section}

## Key Business Findings

- **Revenue concentration:** the **{top_category}** category drives
  **{top_category_pct}%** of total line-item revenue, while **{worst_category}**
  contributes only **{worst_category_pct}%** — a meaningful gap worth
  investigating for assortment planning.
- **Regional spread:** **{top_region}** is the top-performing region by
  revenue, while **{worst_region}** trails the others; see the regional
  breakdown below.
- **Customer risk distribution:** of the scored customer population,
  {churn_summary['high_risk_count']:,} are high-risk, {churn_summary['medium_risk_count']:,}
  are medium-risk, and {churn_summary['low_risk_count']:,} are low-risk
  (average predicted churn probability: {churn_summary['avg_churn_probability']:.1%}).
- **Top churn drivers (SHAP):**
{shap_section}

## Customer Segment Overview

| Segment | Customers | % of Base | Avg Recency (days) | Avg Frequency | Avg Monetary ($) |
|---|---|---|---|---|---|
{chr(10).join(
    f"| {seg} | {int(row['customers'])} | {row['pct_of_base']}% | {row['avg_recency_days']} | {row['avg_frequency']} | {row['avg_monetary']:.2f} |"
    for seg, row in segment_summary.iterrows()
)}

## Customer Retention Recommendations

1. **Prioritize by churn probability, not just segment.** Cross-reference
   `data/processed/churn_predictions.csv` with `data/processed/customer_segments.csv`
   on `customer_id` — a customer can be nominally "Loyal" but still show a
   rising churn probability, and that combination should be the top
   retention priority.
2. **Reserve high-cost incentives for high-value, high-risk customers.**
   Blanket discounting is inefficient; target customers where
   `monetary` is above the segment average **and** `churn_probability >= 0.6`.
3. **Automate low-cost win-back nudges for low-value, low-recency customers**
   rather than spending premium retention budget on them (see Hibernating
   segment below).

## Segment-Specific Actions

"""

    if at_risk_row is not None:
        content += (
            f"### At Risk ({int(at_risk_row['customers'])} customers, "
            f"{at_risk_row['pct_of_base']}% of base)\n"
            f"Average recency is {at_risk_row['avg_recency_days']} days and average historical "
            f"spend is ${at_risk_row['avg_monetary']:.2f} — these customers have real purchase "
            f"history but are drifting. Run a win-back campaign referencing their previously "
            f"purchased categories before they fully churn.\n\n"
        )

    if hibernating_row is not None:
        content += (
            f"### Hibernating ({int(hibernating_row['customers'])} customers, "
            f"{hibernating_row['pct_of_base']}% of base)\n"
            f"Average recency is {hibernating_row['avg_recency_days']} days with average spend of "
            f"only ${hibernating_row['avg_monetary']:.2f}. This segment is the least likely to "
            f"respond to expensive incentives — use low-cost automated email re-engagement only.\n\n"
        )

    for seg in segment_summary.index:
        if seg in ("At Risk", "Hibernating"):
            continue
        row = segment_summary.loc[seg]
        content += (
            f"### {seg} ({int(row['customers'])} customers, {row['pct_of_base']}% of base)\n"
            f"Average recency {row['avg_recency_days']} days, average frequency "
            f"{row['avg_frequency']} orders, average spend ${row['avg_monetary']:.2f}. "
            f"See `reports/segment_rfm_characteristics.png` for a visual comparison against "
            f"other segments.\n\n"
        )

    content += f"""## Churn Risk Actions

- **{churn_summary['high_risk_count']:,} customers** are currently flagged high-risk
  (churn probability ≥ 70%) in `data/processed/churn_predictions.csv`. Export this
  list weekly and route it to the retention/CRM team.
- **{churn_summary['medium_risk_count']:,} customers** are medium-risk (40–70%) — good
  candidates for lighter-touch retention (email/loyalty nudges) rather than
  high-cost intervention.
- Use `reports/shap_summary_plot.png` and `reports/shap_local_explanation_example.png`
  to explain *why* any individual customer is flagged, so retention agents can
  personalize outreach instead of sending generic messaging.

## Product / Category Opportunities

| Category | Revenue ($) | % of Revenue | Units Sold | Avg Discount |
|---|---|---|---|---|
{chr(10).join(
    f"| {cat} | {row['revenue']:,.2f} | {row['pct_of_revenue']}% | {int(row['units_sold'])} | {row['avg_discount']:.1%} |"
    for cat, row in category_perf.iterrows()
)}

**Bottom 5 products by revenue** (candidates for markdown, bundling, or
discontinuation review):

| Product | Category | Revenue ($) | Avg Discount |
|---|---|---|---|
{chr(10).join(
    f"| {row['product_name']} | {row['category']} | {row['revenue']:,.2f} | {row['avg_discount']:.1%} |"
    for _, row in bottom_products.iterrows()
)}

## Regional Opportunities

| Region | Revenue ($) | Orders | Customers | Avg Order Value ($) |
|---|---|---|---|---|
{chr(10).join(
    f"| {region} | {row['revenue']:,.2f} | {int(row['orders'])} | {int(row['customers'])} | {row['avg_order_value']:.2f} |"
    for region, row in regional_perf.iterrows()
)}

**{top_region}** is the strongest region by revenue; **{worst_region}** has the
most room to grow. Compare regional category mix using
`sql/business_analysis.sql` query #7 (Regional Performance) to decide whether
{worst_region}'s gap is driven by assortment, pricing, or customer mix.

## Recommended Next Steps

1. Automate a weekly export of `churn_probability >= 0.6` customers from
   `data/processed/churn_predictions.csv` (or `dashboard/sample_data/churn_predictions.csv`)
   into the retention team's workflow.
2. Track segment migration month over month by re-running
   `python -m src.customer_segmentation` on a rolling basis and diffing
   `customer_segments.csv` snapshots.
3. A/B test retention offers specifically on the **At Risk** segment, since
   it combines proven purchase history with rising disengagement — the
   highest-leverage group for incremental recovery.
4. Review the bottom-performing products listed above for markdown or
   bundling, especially where `avg_discount` is already high without
   translating into meaningful revenue.
5. Investigate the regional revenue gap between {top_region} and {worst_region}
   using the category-by-region cross-tab available via
   `sql/business_analysis.sql`.
"""

    return content


def run() -> None:
    data = load_inputs()
    content = render_markdown(data)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "business_recommendations.md"
    out_path.write_text(content, encoding="utf-8")
    logger.info("Saved %s", out_path)

def main() -> int:
    logger.info("=" * 70)
    logger.info("OmniStyle Customer Intelligence Platform - Business Recommendations")
    logger.info("=" * 70)
    try:
        run()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
