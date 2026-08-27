"""
src/feature_engineering.py
============================

Builds customer-level features, an RFM (Recency/Frequency/Monetary) dataset,
and a leakage-free churn dataset from the cleaned data in data/processed/.

CHURN DEFINITION (documented explicitly to avoid ambiguity / leakage)
-----------------------------------------------------------------------
We use a fixed **cutoff date** that splits history into:

  - Feature window:      all orders with order_date <= CUTOFF_DATE
                          (used to compute recency/frequency/monetary/etc.)
  - Observation window:  CUTOFF_DATE < order_date <= CUTOFF_DATE + 90 days
                          (used ONLY to determine the churn label)

A customer is labeled **churned (1)** if they made **zero purchases** during
the 90-day observation window following the cutoff, **and had signed up
before the cutoff date** (so we only evaluate customers who had a chance to
be active). Customers are labeled **not churned (0)** if they made at least
one purchase during the observation window.

Because features are computed strictly from data up to CUTOFF_DATE, and the
label is computed strictly from data after CUTOFF_DATE, there is no leakage
of future information into the feature set.

CUTOFF_DATE is set 90 days before the project's REFERENCE_DATE, so that a
full 90-day observation window is available within the synthetic dataset's
date range.

Run as:
    python -m src.feature_engineering
"""

from __future__ import annotations

import sys
from datetime import timedelta

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, REFERENCE_DATE_STR, get_logger

logger = get_logger(__name__)

REFERENCE_DATE = pd.Timestamp(REFERENCE_DATE_STR)
CUTOFF_DATE = REFERENCE_DATE - pd.Timedelta(days=90)
OBSERVATION_WINDOW_DAYS = 90
OBSERVATION_END_DATE = CUTOFF_DATE + pd.Timedelta(days=OBSERVATION_WINDOW_DAYS)

LOYALTY_TIER_ENCODING = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}


def _load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = ["customers_clean.csv", "products_clean.csv", "orders_clean.csv", "order_items_clean.csv"]
    missing = [f for f in required if not (DATA_PROCESSED_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed file(s): {missing}. Run 'python -m src.data_cleaning' first."
        )

    customers = pd.read_csv(DATA_PROCESSED_DIR / "customers_clean.csv", parse_dates=["signup_date"])
    products = pd.read_csv(DATA_PROCESSED_DIR / "products_clean.csv")
    orders = pd.read_csv(DATA_PROCESSED_DIR / "orders_clean.csv", parse_dates=["order_date"])
    order_items = pd.read_csv(DATA_PROCESSED_DIR / "order_items_clean.csv")
    return customers, products, orders, order_items


def build_rfm_features(customers: pd.DataFrame, orders: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary using only orders up to `as_of`."""
    orders_window = orders[orders["order_date"] <= as_of]

    agg = orders_window.groupby("customer_id").agg(
        frequency=("order_id", "count"),
        monetary=("total_amount", "sum"),
        last_purchase_date=("order_date", "max"),
        first_purchase_date=("order_date", "min"),
    ).reset_index()

    rfm = customers[["customer_id", "signup_date"]].merge(agg, on="customer_id", how="left")

    # Customers with no purchases in the window get frequency/monetary = 0
    # and recency measured from signup (maximally "stale").
    rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    rfm["monetary"] = rfm["monetary"].fillna(0.0)

    rfm["recency_days"] = (as_of - rfm["last_purchase_date"]).dt.days
    no_purchase_mask = rfm["last_purchase_date"].isna()
    rfm.loc[no_purchase_mask, "recency_days"] = (as_of - rfm.loc[no_purchase_mask, "signup_date"]).dt.days
    rfm["recency_days"] = rfm["recency_days"].clip(lower=0)

    rfm["avg_order_value"] = np.where(rfm["frequency"] > 0, rfm["monetary"] / rfm["frequency"], 0.0)

    return rfm[["customer_id", "recency_days", "frequency", "monetary", "avg_order_value"]]


def _rfm_score(series: pd.Series, ascending: bool) -> pd.Series:
    """Score a series into quintiles 1-5. Handles low-cardinality edge cases gracefully."""
    try:
        ranks = pd.qcut(series.rank(method="first"), 5, labels=False, duplicates="drop") + 1
    except ValueError:
        ranks = pd.Series(1, index=series.index)
    if not ascending:
        max_rank = ranks.max()
        ranks = (max_rank + 1) - ranks
    return ranks.astype(int)


def add_rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    # Lower recency (more recent) = better score (5). Higher frequency/monetary = better.
    rfm["R_score"] = _rfm_score(rfm["recency_days"], ascending=False)
    rfm["F_score"] = _rfm_score(rfm["frequency"], ascending=True)
    rfm["M_score"] = _rfm_score(rfm["monetary"], ascending=True)
    rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
    return rfm


def build_customer_features(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Build the full customer-level feature table using data up to `as_of`."""
    orders_window = orders[orders["order_date"] <= as_of].copy()
    order_ids_in_window = set(orders_window["order_id"])
    items_window = order_items[order_items["order_id"].isin(order_ids_in_window)].copy()

    items_window = items_window.merge(
        orders_window[["order_id", "customer_id"]], on="order_id", how="left"
    ).merge(products[["product_id", "category"]], on="product_id", how="left")

    rfm = build_rfm_features(customers, orders, as_of)

    tenure = customers[["customer_id", "signup_date"]].copy()
    tenure["customer_tenure_days"] = (as_of - tenure["signup_date"]).dt.days.clip(lower=0)

    items_agg = items_window.groupby("customer_id").agg(
        total_items_purchased=("quantity", "sum"),
        unique_products_purchased=("product_id", "nunique"),
        unique_categories_purchased=("category", "nunique"),
        average_discount=("discount", "mean"),
    ).reset_index()

    # Purchase trend: compare spend in the most recent 90 days of the window
    # vs. the prior 90 days, as a simple momentum indicator (>1 = accelerating).
    recent_start = as_of - pd.Timedelta(days=90)
    prior_start = as_of - pd.Timedelta(days=180)

    recent_spend = orders_window[orders_window["order_date"] > recent_start].groupby("customer_id")["total_amount"].sum()
    prior_spend = orders_window[
        (orders_window["order_date"] > prior_start) & (orders_window["order_date"] <= recent_start)
    ].groupby("customer_id")["total_amount"].sum()

    trend_df = pd.DataFrame({"recent_90d_spend": recent_spend, "prior_90d_spend": prior_spend}).fillna(0.0)
    trend_df["purchase_trend_ratio"] = np.where(
        trend_df["prior_90d_spend"] > 0,
        trend_df["recent_90d_spend"] / trend_df["prior_90d_spend"],
        np.where(trend_df["recent_90d_spend"] > 0, 2.0, 1.0),
    )
    trend_df = trend_df.reset_index()

    features = (
        customers[["customer_id", "loyalty_tier", "loyalty_points", "region", "age", "gender"]]
        .merge(tenure[["customer_id", "customer_tenure_days"]], on="customer_id", how="left")
        .merge(rfm, on="customer_id", how="left")
        .merge(items_agg, on="customer_id", how="left")
        .merge(trend_df[["customer_id", "purchase_trend_ratio"]], on="customer_id", how="left")
    )

    fill_zero_cols = [
        "total_items_purchased", "unique_products_purchased", "unique_categories_purchased",
        "average_discount", "frequency", "monetary", "avg_order_value",
    ]
    for col in fill_zero_cols:
        features[col] = features[col].fillna(0)

    features["purchase_trend_ratio"] = features["purchase_trend_ratio"].fillna(1.0)
    features["loyalty_tier_encoded"] = features["loyalty_tier"].map(LOYALTY_TIER_ENCODING).fillna(0).astype(int)

    return features


def build_churn_labels(orders: pd.DataFrame, customers: pd.DataFrame, cutoff: pd.Timestamp, obs_end: pd.Timestamp) -> pd.DataFrame:
    """
    Label churn using a strict, leakage-free cutoff:
      - Only customers who signed up on/before `cutoff` are eligible for labeling
        (they had a genuine chance to purchase in the observation window).
      - churn = 1 if the customer has NO orders in (cutoff, obs_end].
      - churn = 0 if the customer has >=1 order in (cutoff, obs_end].
    """
    eligible = customers[customers["signup_date"] <= cutoff][["customer_id"]].copy()

    future_orders = orders[(orders["order_date"] > cutoff) & (orders["order_date"] <= obs_end)]
    active_in_window = set(future_orders["customer_id"].unique())

    eligible["churned"] = (~eligible["customer_id"].isin(active_in_window)).astype(int)
    return eligible


def run_feature_engineering_pipeline() -> pd.DataFrame:
    customers, products, orders, order_items = _load_processed()

    logger.info("Reference date: %s | Cutoff date: %s | Observation window ends: %s",
                REFERENCE_DATE.date(), CUTOFF_DATE.date(), OBSERVATION_END_DATE.date())

    rfm = build_rfm_features(customers, orders, as_of=CUTOFF_DATE)
    rfm = add_rfm_scores(rfm)
    rfm_out = customers[["customer_id"]].merge(rfm, on="customer_id", how="left")
    rfm_out.to_csv(DATA_PROCESSED_DIR / "rfm_features.csv", index=False)
    logger.info("Saved rfm_features.csv (%d rows)", len(rfm_out))

    features = build_customer_features(customers, orders, order_items, products, as_of=CUTOFF_DATE)
    features.to_csv(DATA_PROCESSED_DIR / "customer_features.csv", index=False)
    logger.info("Saved customer_features.csv (%d rows, %d columns)", *features.shape)

    labels = build_churn_labels(orders, customers, cutoff=CUTOFF_DATE, obs_end=OBSERVATION_END_DATE)
    churn_dataset = features.merge(labels, on="customer_id", how="inner")
    churn_dataset.to_csv(DATA_PROCESSED_DIR / "churn_dataset.csv", index=False)

    churn_rate = churn_dataset["churned"].mean()
    logger.info(
        "Saved churn_dataset.csv (%d eligible customers, churn rate = %.1f%%)",
        len(churn_dataset), churn_rate * 100,
    )

    return churn_dataset


def main() -> int:
    logger.info("=" * 70)
    logger.info("OmniStyle Customer Intelligence Platform - Feature Engineering")
    logger.info("=" * 70)
    run_feature_engineering_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
