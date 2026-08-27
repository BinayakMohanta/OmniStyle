"""
Unit tests for src/feature_engineering.py

Uses small, hand-crafted DataFrames to validate RFM computation and the
leakage-free churn labeling logic in isolation.
"""

import pandas as pd

from src.feature_engineering import (
    build_churn_labels,
    build_customer_features,
    build_rfm_features,
)


def _sample_customers():
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "signup_date": pd.to_datetime(["2023-01-01", "2023-06-01", "2024-01-01"]),
            "loyalty_tier": ["Gold", "Silver", "Bronze"],
            "loyalty_points": [1000, 500, 100],
            "region": ["East", "West", "North"],
            "age": [30, 40, 25],
            "gender": ["Female", "Male", "Female"],
        }
    )


def _sample_orders():
    return pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3", "O4"],
            "customer_id": ["C1", "C1", "C2", "C1"],
            "order_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-15", "2024-06-01"]),
            "total_amount": [100.0, 50.0, 200.0, 75.0],
            "payment_method": ["Credit Card"] * 4,
            "channel": ["Online"] * 4,
            "store_region": ["East", "East", "West", "East"],
        }
    )


def test_rfm_frequency_and_monetary_counts_only_orders_up_to_cutoff():
    customers = _sample_customers()
    orders = _sample_orders()
    as_of = pd.Timestamp("2024-03-01")  # excludes O4 (2024-06-01)

    rfm = build_rfm_features(customers, orders, as_of=as_of)
    c1 = rfm[rfm["customer_id"] == "C1"].iloc[0]

    assert c1["frequency"] == 2  # O1 and O2 only, O4 is after cutoff
    assert c1["monetary"] == 150.0  # 100 + 50


def test_rfm_customer_with_no_orders_has_zero_frequency():
    customers = _sample_customers()
    orders = _sample_orders()
    as_of = pd.Timestamp("2024-03-01")

    rfm = build_rfm_features(customers, orders, as_of=as_of)
    c3 = rfm[rfm["customer_id"] == "C3"].iloc[0]

    assert c3["frequency"] == 0
    assert c3["monetary"] == 0.0
    assert c3["recency_days"] >= 0


def test_churn_label_marks_customer_with_no_future_orders_as_churned():
    customers = _sample_customers()
    orders = _sample_orders()

    cutoff = pd.Timestamp("2024-02-15")
    obs_end = cutoff + pd.Timedelta(days=90)  # covers up to 2024-05-15

    labels = build_churn_labels(orders, customers, cutoff=cutoff, obs_end=obs_end)

    # C2's only order (O3, 2024-01-15) is before cutoff -> no orders in window -> churned.
    c2_label = labels.loc[labels["customer_id"] == "C2", "churned"].iloc[0]
    assert c2_label == 1


def test_churn_label_marks_customer_with_future_order_as_not_churned():
    customers = _sample_customers()
    orders = _sample_orders()

    cutoff = pd.Timestamp("2024-02-15")
    obs_end = cutoff + pd.Timedelta(days=120)  # covers up to O4 (2024-06-01)

    labels = build_churn_labels(orders, customers, cutoff=cutoff, obs_end=obs_end)

    # C1 has O4 (2024-06-01) which falls inside (cutoff, obs_end] -> not churned.
    c1_label = labels.loc[labels["customer_id"] == "C1", "churned"].iloc[0]
    assert c1_label == 0


def test_churn_label_excludes_customers_who_signed_up_after_cutoff():
    customers = _sample_customers()
    orders = _sample_orders()

    # C3 signed up 2024-01-01; use a cutoff before that so C3 is ineligible.
    cutoff = pd.Timestamp("2023-12-01")
    obs_end = cutoff + pd.Timedelta(days=90)

    labels = build_churn_labels(orders, customers, cutoff=cutoff, obs_end=obs_end)
    assert "C3" not in set(labels["customer_id"])


def test_customer_features_output_has_expected_columns_and_no_nulls():
    customers = _sample_customers()
    orders = _sample_orders()
    order_items = pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3", "O4"],
            "product_id": ["P1", "P2", "P1", "P3"],
            "quantity": [1, 2, 1, 1],
            "unit_price": [100.0, 25.0, 200.0, 75.0],
            "discount": [0.0, 0.1, 0.0, 0.05],
            "line_total": [100.0, 45.0, 200.0, 71.25],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["P1", "P2", "P3"],
            "category": ["Men", "Women", "Kids"],
        }
    )

    as_of = pd.Timestamp("2024-03-01")
    features = build_customer_features(customers, orders, order_items, products, as_of=as_of)

    expected_cols = {
        "customer_id", "loyalty_tier", "loyalty_points", "region", "age", "gender",
        "customer_tenure_days", "recency_days", "frequency", "monetary", "avg_order_value",
        "total_items_purchased", "unique_products_purchased", "unique_categories_purchased",
        "average_discount", "purchase_trend_ratio", "loyalty_tier_encoded",
    }
    assert expected_cols.issubset(set(features.columns))
    assert features["customer_id"].notna().all()
    assert not features[["frequency", "monetary", "total_items_purchased"]].isna().any().any()
