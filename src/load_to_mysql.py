"""
src/load_to_mysql.py
======================

Loads the cleaned, processed CSV datasets into the MySQL star-schema
analytics database defined in sql/schema.sql.

Prerequisites:
  1. MySQL server running and reachable using the MYSQL_* env vars.
  2. sql/schema.sql already applied:
         mysql -u <user> -p < sql/schema.sql

Run as:
    python -m src.load_to_mysql
"""

from __future__ import annotations

import sys

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DATA_PROCESSED_DIR, get_logger, get_mysql_uri

logger = get_logger(__name__)


def _build_dim_date(orders: pd.DataFrame) -> pd.DataFrame:
    min_date = orders["order_date"].min()
    max_date = orders["order_date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="D")

    dim_date = pd.DataFrame({"date_key": all_dates})
    dim_date["year"] = dim_date["date_key"].dt.year
    dim_date["quarter"] = dim_date["date_key"].dt.quarter
    dim_date["month"] = dim_date["date_key"].dt.month
    dim_date["month_name"] = dim_date["date_key"].dt.month_name()
    dim_date["day"] = dim_date["date_key"].dt.day
    dim_date["day_of_week"] = dim_date["date_key"].dt.dayofweek
    dim_date["day_name"] = dim_date["date_key"].dt.day_name()
    dim_date["is_weekend"] = dim_date["day_of_week"].isin([5, 6]).astype(int)
    return dim_date


def load_all() -> None:
    logger.info("Loading processed CSVs from %s", DATA_PROCESSED_DIR)
    customers = pd.read_csv(DATA_PROCESSED_DIR / "customers_clean.csv", parse_dates=["signup_date"])
    products = pd.read_csv(DATA_PROCESSED_DIR / "products_clean.csv")
    orders = pd.read_csv(DATA_PROCESSED_DIR / "orders_clean.csv", parse_dates=["order_date"])
    order_items = pd.read_csv(DATA_PROCESSED_DIR / "order_items_clean.csv")

    dim_date = _build_dim_date(orders)

    fact_sales = order_items.merge(
        orders[["order_id", "customer_id", "order_date", "payment_method", "channel", "store_region"]],
        on="order_id", how="inner",
    )
    fact_sales = fact_sales.rename(columns={"order_date": "date_key"})
    fact_sales = fact_sales[
        ["order_id", "customer_id", "product_id", "date_key", "quantity",
         "unit_price", "discount", "line_total", "payment_method", "channel", "store_region"]
    ]

    engine = create_engine(get_mysql_uri())

    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        conn.execute(text("TRUNCATE TABLE fact_sales"))
        conn.execute(text("TRUNCATE TABLE dim_date"))
        conn.execute(text("TRUNCATE TABLE dim_product"))
        conn.execute(text("TRUNCATE TABLE dim_customer"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))

    logger.info("Loading dim_customer (%d rows)...", len(customers))
    customers.to_sql("dim_customer", engine, if_exists="append", index=False, chunksize=1000)

    logger.info("Loading dim_product (%d rows)...", len(products))
    products.to_sql("dim_product", engine, if_exists="append", index=False, chunksize=1000)

    logger.info("Loading dim_date (%d rows)...", len(dim_date))
    dim_date.to_sql("dim_date", engine, if_exists="append", index=False, chunksize=1000)

    logger.info("Loading fact_sales (%d rows)...", len(fact_sales))
    fact_sales.to_sql("fact_sales", engine, if_exists="append", index=False, chunksize=5000)

    logger.info("MySQL load complete.")


def main() -> int:
    logger.info("=" * 70)
    logger.info("OmniStyle Customer Intelligence Platform - MySQL Load")
    logger.info("=" * 70)
    try:
        load_all()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load data into MySQL: %s", exc)
        logger.error(
            "Ensure MySQL is running, credentials in .env are correct, and "
            "sql/schema.sql has been applied first: mysql -u <user> -p < sql/schema.sql"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
