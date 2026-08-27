"""
src/data_cleaning.py
======================

Loads raw CSV exports from data/raw/, validates and cleans them, and writes
clean datasets plus a data-quality report to data/processed/.

Run as:
    python -m src.data_cleaning
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, get_logger

logger = get_logger(__name__)

REQUIRED_RAW_FILES = ["customers.csv", "products.csv", "orders.csv", "order_items.csv"]


class DataQualityTracker:
    """Accumulates data-quality issues found during cleaning for the final report."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def log(self, dataset: str, check: str, issue_count: int, action: str) -> None:
        self.records.append(
            {"dataset": dataset, "check": check, "issue_count": int(issue_count), "action_taken": action}
        )
        if issue_count > 0:
            logger.info("[%s] %s: %d issue(s) -> %s", dataset, check, issue_count, action)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def _check_files_exist() -> None:
    missing = [f for f in REQUIRED_RAW_FILES if not (DATA_RAW_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw data file(s): {missing} in {DATA_RAW_DIR}.\n"
            f"Run 'python -m src.data_ingestion' first to generate raw data."
        )


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _check_files_exist()
    logger.info("Loading raw data from %s", DATA_RAW_DIR)
    customers = pd.read_csv(DATA_RAW_DIR / "customers.csv")
    products = pd.read_csv(DATA_RAW_DIR / "products.csv")
    orders = pd.read_csv(DATA_RAW_DIR / "orders.csv")
    order_items = pd.read_csv(DATA_RAW_DIR / "order_items.csv")
    logger.info(
        "Loaded: customers=%d, products=%d, orders=%d, order_items=%d",
        len(customers), len(products), len(orders), len(order_items),
    )
    return customers, products, orders, order_items


def clean_customers(df: pd.DataFrame, tracker: DataQualityTracker) -> pd.DataFrame:
    df = df.copy()

    dup_count = df.duplicated(subset="customer_id").sum()
    df = df.drop_duplicates(subset="customer_id", keep="first")
    tracker.log("customers", "duplicate_customer_id", dup_count, "dropped duplicates, kept first occurrence")

    missing_id = df["customer_id"].isna().sum()
    df = df[df["customer_id"].notna()]
    tracker.log("customers", "missing_customer_id", missing_id, "dropped rows with missing customer_id")

    for col in ["name", "email", "city", "state", "region", "loyalty_tier"]:
        missing = df[col].isna().sum()
        if missing > 0:
            df[col] = df[col].fillna("Unknown")
        tracker.log("customers", f"missing_{col}", missing, "filled with 'Unknown'" if missing else "none found")

    invalid_age = ((df["age"] < 16) | (df["age"] > 100) | df["age"].isna()).sum()
    median_age = df.loc[(df["age"] >= 16) & (df["age"] <= 100), "age"].median()
    df.loc[(df["age"] < 16) | (df["age"] > 100) | df["age"].isna(), "age"] = median_age
    df["age"] = df["age"].astype(int)
    tracker.log("customers", "invalid_age", invalid_age, f"replaced with median age ({median_age:.0f})")

    invalid_points = (df["loyalty_points"] < 0).sum()
    df.loc[df["loyalty_points"] < 0, "loyalty_points"] = 0
    tracker.log("customers", "negative_loyalty_points", invalid_points, "clipped to 0")

    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    bad_dates = df["signup_date"].isna().sum()
    if bad_dates > 0:
        df = df[df["signup_date"].notna()]
    tracker.log("customers", "invalid_signup_date", bad_dates, "dropped rows with unparseable signup_date")

    valid_tiers = {"Bronze", "Silver", "Gold", "Platinum"}
    invalid_tier = (~df["loyalty_tier"].isin(valid_tiers)).sum()
    df.loc[~df["loyalty_tier"].isin(valid_tiers), "loyalty_tier"] = "Bronze"
    tracker.log("customers", "invalid_loyalty_tier", invalid_tier, "defaulted to 'Bronze'")

    return df.reset_index(drop=True)


def clean_products(df: pd.DataFrame, tracker: DataQualityTracker) -> pd.DataFrame:
    df = df.copy()

    dup_count = df.duplicated(subset="product_id").sum()
    df = df.drop_duplicates(subset="product_id", keep="first")
    tracker.log("products", "duplicate_product_id", dup_count, "dropped duplicates, kept first occurrence")

    missing_id = df["product_id"].isna().sum()
    df = df[df["product_id"].notna()]
    tracker.log("products", "missing_product_id", missing_id, "dropped rows with missing product_id")

    invalid_price = ((df["unit_price"] <= 0) | df["unit_price"].isna()).sum()
    median_price = df.loc[df["unit_price"] > 0, "unit_price"].median()
    df.loc[(df["unit_price"] <= 0) | df["unit_price"].isna(), "unit_price"] = median_price
    tracker.log("products", "invalid_unit_price", invalid_price, f"replaced with median price ({median_price:.2f})")

    invalid_cost = ((df["unit_cost"] <= 0) | df["unit_cost"].isna() | (df["unit_cost"] >= df["unit_price"])).sum()
    df.loc[
        (df["unit_cost"] <= 0) | df["unit_cost"].isna() | (df["unit_cost"] >= df["unit_price"]),
        "unit_cost",
    ] = df["unit_price"] * 0.6
    tracker.log("products", "invalid_unit_cost", invalid_cost, "recomputed as 60% of unit_price")

    return df.reset_index(drop=True)


def clean_orders_and_items(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    valid_customer_ids: set,
    valid_product_ids: set,
    tracker: DataQualityTracker,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = orders.copy()
    items = items.copy()

    dup_orders = orders.duplicated(subset="order_id").sum()
    orders = orders.drop_duplicates(subset="order_id", keep="first")
    tracker.log("orders", "duplicate_order_id", dup_orders, "dropped duplicates, kept first occurrence")

    orphan_customers = (~orders["customer_id"].isin(valid_customer_ids)).sum()
    orders = orders[orders["customer_id"].isin(valid_customer_ids)]
    tracker.log("orders", "orphan_customer_reference", orphan_customers, "dropped orders referencing unknown customers")

    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
    bad_dates = orders["order_date"].isna().sum()
    orders = orders[orders["order_date"].notna()]
    tracker.log("orders", "invalid_order_date", bad_dates, "dropped rows with unparseable order_date")

    negative_totals = (orders["total_amount"] < 0).sum()
    orders = orders[orders["total_amount"] >= 0]
    tracker.log("orders", "negative_total_amount", negative_totals, "dropped rows with negative total_amount")

    valid_order_ids = set(orders["order_id"])

    orphan_order_ref = (~items["order_id"].isin(valid_order_ids)).sum()
    items = items[items["order_id"].isin(valid_order_ids)]
    tracker.log("order_items", "orphan_order_reference", orphan_order_ref, "dropped items referencing unknown/removed orders")

    orphan_product_ref = (~items["product_id"].isin(valid_product_ids)).sum()
    items = items[items["product_id"].isin(valid_product_ids)]
    tracker.log("order_items", "orphan_product_reference", orphan_product_ref, "dropped items referencing unknown products")

    invalid_qty = ((items["quantity"] <= 0) | items["quantity"].isna()).sum()
    items = items[(items["quantity"] > 0) & items["quantity"].notna()]
    tracker.log("order_items", "invalid_quantity", invalid_qty, "dropped rows with non-positive quantity")

    invalid_line_total = ((items["line_total"] < 0) | items["line_total"].isna()).sum()
    items = items[(items["line_total"] >= 0) & items["line_total"].notna()]
    tracker.log("order_items", "invalid_line_total", invalid_line_total, "dropped rows with negative/missing line_total")

    invalid_discount = ((items["discount"] < 0) | (items["discount"] > 0.9)).sum()
    items["discount"] = items["discount"].clip(lower=0, upper=0.9)
    tracker.log("order_items", "invalid_discount_range", invalid_discount, "clipped discount to [0, 0.9]")

    # Cross-validate order total_amount against the sum of its line items.
    items_recalc = items.groupby("order_id")["line_total"].sum().round(2)
    orders = orders.set_index("order_id")
    orders["computed_total"] = items_recalc
    orders["computed_total"] = orders["computed_total"].fillna(0.0)
    mismatch_mask = (orders["total_amount"] - orders["computed_total"]).abs() > 1.0
    mismatch_count = mismatch_mask.sum()
    # Trust the line-item detail as the source of truth and correct the header total.
    orders.loc[mismatch_mask, "total_amount"] = orders.loc[mismatch_mask, "computed_total"]
    orders = orders.drop(columns=["computed_total"]).reset_index()
    tracker.log(
        "orders", "total_amount_mismatch_vs_line_items", mismatch_count,
        "recalculated total_amount from order_items line totals",
    )

    # Drop orders that ended up with zero valid line items after cleaning.
    orders_with_items = set(items["order_id"])
    empty_orders = (~orders["order_id"].isin(orders_with_items)).sum()
    orders = orders[orders["order_id"].isin(orders_with_items)]
    tracker.log("orders", "orders_with_no_valid_items", empty_orders, "dropped orders with zero remaining valid line items")

    return orders.reset_index(drop=True), items.reset_index(drop=True)


def run_cleaning_pipeline() -> pd.DataFrame:
    tracker = DataQualityTracker()

    customers, products, orders, order_items = load_raw_data()

    customers_clean = clean_customers(customers, tracker)
    products_clean = clean_products(products, tracker)
    orders_clean, order_items_clean = clean_orders_and_items(
        orders, order_items,
        valid_customer_ids=set(customers_clean["customer_id"]),
        valid_product_ids=set(products_clean["product_id"]),
        tracker=tracker,
    )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    customers_clean.to_csv(DATA_PROCESSED_DIR / "customers_clean.csv", index=False)
    products_clean.to_csv(DATA_PROCESSED_DIR / "products_clean.csv", index=False)
    orders_clean.to_csv(DATA_PROCESSED_DIR / "orders_clean.csv", index=False)
    order_items_clean.to_csv(DATA_PROCESSED_DIR / "order_items_clean.csv", index=False)

    quality_report = tracker.to_dataframe()
    quality_report.to_csv(DATA_PROCESSED_DIR / "data_quality_report.csv", index=False)

    logger.info(
        "Cleaning complete. Final row counts: customers=%d, products=%d, orders=%d, order_items=%d",
        len(customers_clean), len(products_clean), len(orders_clean), len(order_items_clean),
    )
    total_issues = quality_report["issue_count"].sum() if not quality_report.empty else 0
    logger.info("Total data quality issues found and handled: %d", total_issues)
    logger.info("Data quality report saved to %s", DATA_PROCESSED_DIR / "data_quality_report.csv")

    return quality_report


def main() -> int:
    logger.info("=" * 70)
    logger.info("OmniStyle Customer Intelligence Platform - Data Cleaning")
    logger.info("=" * 70)
    run_cleaning_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
