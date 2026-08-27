"""
src/export_dashboard_data.py
==============================

Standalone Power BI data export module.

`src/train_model.py` already writes a first set of Power BI-ready CSVs
(`sales_detail.csv`, `customers.csv`, `products.csv`, `orders.csv`,
`customer_360.csv`) as part of its combined pipeline — that behavior is
preserved unchanged.

This module is a **standalone, independently runnable** exporter that reads
only from already-generated files on disk (it does not require being called
from within `train_model.py`), and produces the additional business-friendly
exports requested for the dashboard:

- sales_dashboard.csv   -- one row per order line item, with clean,
                           business-friendly column names and derived date parts
- customer_dashboard.csv -- one row per customer: profile + RFM + segment + churn risk
- customer_segments.csv -- copy of the segmentation output, for direct Power BI import
- churn_predictions.csv -- copy of the churn scoring output, for direct Power BI import
- dim_date.csv          -- a calendar date dimension spanning the order history
                           (reuses the same logic as src/load_to_mysql.py so the
                           Power BI date dimension matches the MySQL one)

All data comes from real files already produced earlier in the pipeline
(`data/processed/*.csv`) — nothing here is fabricated.

Run as:
    python -m src.export_dashboard_data
"""

from __future__ import annotations

import sys

import pandas as pd

from src.config import DASHBOARD_SAMPLE_DATA_DIR, DATA_PROCESSED_DIR, get_logger
from src.load_to_mysql import _build_dim_date

logger = get_logger(__name__)

REQUIRED_FILES = [
    "customers_clean.csv",
    "products_clean.csv",
    "orders_clean.csv",
    "order_items_clean.csv",
    "customer_features.csv",
]


def _check_prerequisites() -> None:
    missing = [f for f in REQUIRED_FILES if not (DATA_PROCESSED_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed file(s): {missing} in {DATA_PROCESSED_DIR}.\n"
            f"Run the earlier pipeline stages first: "
            f"'python -m src.data_ingestion', 'python -m src.data_cleaning', "
            f"'python -m src.feature_engineering'."
        )


def load_processed_tables() -> dict[str, pd.DataFrame]:
    _check_prerequisites()
    tables = {
        "customers": pd.read_csv(DATA_PROCESSED_DIR / "customers_clean.csv", parse_dates=["signup_date"]),
        "products": pd.read_csv(DATA_PROCESSED_DIR / "products_clean.csv"),
        "orders": pd.read_csv(DATA_PROCESSED_DIR / "orders_clean.csv", parse_dates=["order_date"]),
        "order_items": pd.read_csv(DATA_PROCESSED_DIR / "order_items_clean.csv"),
        "features": pd.read_csv(DATA_PROCESSED_DIR / "customer_features.csv"),
    }

    segments_path = DATA_PROCESSED_DIR / "customer_segments.csv"
    tables["segments"] = pd.read_csv(segments_path) if segments_path.exists() else None
    if tables["segments"] is None:
        logger.warning(
            "%s not found. Run 'python -m src.customer_segmentation' first for "
            "segment-enriched exports. Continuing without segment data.",
            segments_path,
        )

    churn_path = DATA_PROCESSED_DIR / "churn_predictions.csv"
    tables["churn_predictions"] = pd.read_csv(churn_path) if churn_path.exists() else None
    if tables["churn_predictions"] is None:
        logger.warning(
            "%s not found. Run 'python -m src.train_model' first for "
            "churn-risk-enriched exports. Continuing without churn data.",
            churn_path,
        )

    return tables


def build_sales_dashboard(tables: dict) -> pd.DataFrame:
    """One row per order line item, business-friendly columns, ready for Power BI."""
    order_items = tables["order_items"]
    orders = tables["orders"]
    products = tables["products"]

    sales = (
        order_items.merge(
            orders[["order_id", "customer_id", "order_date", "channel", "store_region", "payment_method"]],
            on="order_id",
            how="inner",
        ).merge(
            products[["product_id", "product_name", "category", "subcategory", "brand", "unit_cost"]],
            on="product_id",
            how="left",
        )
    )

    sales["order_date"] = pd.to_datetime(sales["order_date"])
    sales["order_year"] = sales["order_date"].dt.year
    sales["order_month"] = sales["order_date"].dt.month
    sales["order_month_name"] = sales["order_date"].dt.month_name()
    sales["gross_margin"] = (sales["unit_price"] - sales["unit_cost"]) * sales["quantity"] * (1 - sales["discount"])

    sales = sales.rename(columns={"store_region": "region"})

    column_order = [
        "order_id", "order_date", "order_year", "order_month", "order_month_name",
        "customer_id", "product_id", "product_name", "category", "subcategory", "brand",
        "quantity", "unit_price", "discount", "line_total", "gross_margin",
        "payment_method", "channel", "region",
    ]
    return sales[column_order]


def build_customer_dashboard(tables: dict) -> pd.DataFrame:
    """One row per customer: profile + RFM/behavioral features + segment + churn risk."""
    customers = tables["customers"]
    features = tables["features"]

    dashboard = customers.merge(features, on="customer_id", how="left", suffixes=("", "_dup"))
    # Drop any accidental duplicate columns created by the merge (both frames
    # share loyalty_tier / loyalty_points / region / age / gender).
    dup_cols = [c for c in dashboard.columns if c.endswith("_dup")]
    dashboard = dashboard.drop(columns=dup_cols)

    if tables["segments"] is not None:
        dashboard = dashboard.merge(
            tables["segments"][["customer_id", "segment"]], on="customer_id", how="left"
        )
    else:
        dashboard["segment"] = pd.NA

    if tables["churn_predictions"] is not None:
        dashboard = dashboard.merge(
            tables["churn_predictions"][["customer_id", "churn_probability"]],
            on="customer_id",
            how="left",
        )
    else:
        dashboard["churn_probability"] = pd.NA

    dashboard["churn_risk_band"] = pd.cut(
        dashboard["churn_probability"],
        bins=[-0.01, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    return dashboard


def export_all() -> None:
    logger.info("--- Exporting standalone Power BI dashboard datasets ---")
    tables = load_processed_tables()

    DASHBOARD_SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sales_dashboard = build_sales_dashboard(tables)
    sales_dashboard.to_csv(DASHBOARD_SAMPLE_DATA_DIR / "sales_dashboard.csv", index=False)
    logger.info("Wrote sales_dashboard.csv (%d rows)", len(sales_dashboard))

    customer_dashboard = build_customer_dashboard(tables)
    customer_dashboard.to_csv(DASHBOARD_SAMPLE_DATA_DIR / "customer_dashboard.csv", index=False)
    logger.info("Wrote customer_dashboard.csv (%d rows)", len(customer_dashboard))

    if tables["segments"] is not None:
        tables["segments"].to_csv(DASHBOARD_SAMPLE_DATA_DIR / "customer_segments.csv", index=False)
        logger.info("Wrote customer_segments.csv (%d rows)", len(tables["segments"]))

    if tables["churn_predictions"] is not None:
        tables["churn_predictions"].to_csv(DASHBOARD_SAMPLE_DATA_DIR / "churn_predictions.csv", index=False)
        logger.info("Wrote churn_predictions.csv (%d rows)", len(tables["churn_predictions"]))

    dim_date = _build_dim_date(tables["orders"])
    dim_date.to_csv(DASHBOARD_SAMPLE_DATA_DIR / "dim_date.csv", index=False)
    logger.info("Wrote dim_date.csv (%d rows)", len(dim_date))

    logger.info("Standalone Power BI exports complete: %s", DASHBOARD_SAMPLE_DATA_DIR)


def main() -> int:
    logger.info("=" * 70)
    logger.info("OmniStyle Customer Intelligence Platform - Power BI Export")
    logger.info("=" * 70)
    try:
        export_all()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
