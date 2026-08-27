"""
Unit tests for src/data_cleaning.py

These tests build small, hand-crafted DataFrames (rather than depending on
the full generated dataset) so they run fast and deterministically in CI.
"""

import pandas as pd
import pytest

from src.data_cleaning import (
    DataQualityTracker,
    clean_customers,
    clean_orders_and_items,
    clean_products,
)


@pytest.fixture
def tracker():
    return DataQualityTracker()


def test_clean_customers_removes_duplicates(tracker):
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2"],
            "name": ["Alice", "Alice", "Bob"],
            "email": ["a@x.com", "a@x.com", "b@x.com"],
            "gender": ["Female", "Female", "Male"],
            "age": [30, 30, 40],
            "city": ["NYC", "NYC", "LA"],
            "state": ["NY", "NY", "CA"],
            "region": ["East", "East", "West"],
            "signup_date": ["2023-01-01", "2023-01-01", "2023-02-01"],
            "loyalty_tier": ["Gold", "Gold", "Silver"],
            "loyalty_points": [500, 500, 200],
        }
    )
    cleaned = clean_customers(df, tracker)
    assert len(cleaned) == 2
    assert set(cleaned["customer_id"]) == {"C1", "C2"}


def test_clean_customers_fixes_invalid_age(tracker):
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "name": ["A", "B", "C"],
            "email": ["a@x.com", "b@x.com", "c@x.com"],
            "gender": ["Female", "Male", "Male"],
            "age": [200, -5, 35],  # first two are invalid
            "city": ["NYC", "LA", "SF"],
            "state": ["NY", "CA", "CA"],
            "region": ["East", "West", "West"],
            "signup_date": ["2023-01-01", "2023-02-01", "2023-03-01"],
            "loyalty_tier": ["Gold", "Silver", "Bronze"],
            "loyalty_points": [500, 200, 100],
        }
    )
    cleaned = clean_customers(df, tracker)
    assert (cleaned["age"] == 35).sum() >= 1
    assert cleaned["age"].between(16, 100).all()


def test_clean_customers_negative_loyalty_points_clipped(tracker):
    df = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "name": ["A"],
            "email": ["a@x.com"],
            "gender": ["Female"],
            "age": [30],
            "city": ["NYC"],
            "state": ["NY"],
            "region": ["East"],
            "signup_date": ["2023-01-01"],
            "loyalty_tier": ["Gold"],
            "loyalty_points": [-50],
        }
    )
    cleaned = clean_customers(df, tracker)
    assert cleaned["loyalty_points"].iloc[0] == 0


def test_clean_products_fixes_invalid_price_and_cost(tracker):
    df = pd.DataFrame(
        {
            "product_id": ["P1", "P2"],
            "product_name": ["Shirt", "Jeans"],
            "category": ["Men", "Men"],
            "subcategory": ["Shirts", "Jeans"],
            "brand": ["BrandA", "BrandB"],
            "unit_cost": [-5, 100],   # P1 invalid cost, P2 cost >= price (invalid)
            "unit_price": [20, 50],
        }
    )
    cleaned = clean_products(df, tracker)
    assert (cleaned["unit_cost"] > 0).all()
    assert (cleaned["unit_cost"] < cleaned["unit_price"]).all()


def test_clean_orders_and_items_detects_total_mismatch(tracker):
    orders = pd.DataFrame(
        {
            "order_id": ["O1"],
            "customer_id": ["C1"],
            "order_date": ["2024-01-01"],
            "total_amount": [999.0],  # intentionally wrong vs. line items
            "payment_method": ["Credit Card"],
            "channel": ["Online"],
            "store_region": ["East"],
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["O1"],
            "product_id": ["P1"],
            "quantity": [2],
            "unit_price": [10.0],
            "discount": [0.0],
            "line_total": [20.0],
        }
    )
    orders_clean, items_clean = clean_orders_and_items(
        orders, items, valid_customer_ids={"C1"}, valid_product_ids={"P1"}, tracker=tracker
    )
    assert orders_clean.loc[orders_clean["order_id"] == "O1", "total_amount"].iloc[0] == pytest.approx(20.0)


def test_clean_orders_drops_orphan_references(tracker):
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "UNKNOWN"],
            "order_date": ["2024-01-01", "2024-01-02"],
            "total_amount": [20.0, 15.0],
            "payment_method": ["Credit Card", "UPI"],
            "channel": ["Online", "Online"],
            "store_region": ["East", "West"],
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "product_id": ["P1", "P1"],
            "quantity": [2, 1],
            "unit_price": [10.0, 15.0],
            "discount": [0.0, 0.0],
            "line_total": [20.0, 15.0],
        }
    )
    orders_clean, items_clean = clean_orders_and_items(
        orders, items, valid_customer_ids={"C1"}, valid_product_ids={"P1"}, tracker=tracker
    )
    assert list(orders_clean["order_id"]) == ["O1"]
    assert list(items_clean["order_id"]) == ["O1"]


def test_clean_orders_drops_negative_quantity(tracker):
    orders = pd.DataFrame(
        {
            "order_id": ["O1"],
            "customer_id": ["C1"],
            "order_date": ["2024-01-01"],
            "total_amount": [0.0],
            "payment_method": ["Credit Card"],
            "channel": ["Online"],
            "store_region": ["East"],
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["O1"],
            "product_id": ["P1"],
            "quantity": [-3],
            "unit_price": [10.0],
            "discount": [0.0],
            "line_total": [-30.0],
        }
    )
    orders_clean, items_clean = clean_orders_and_items(
        orders, items, valid_customer_ids={"C1"}, valid_product_ids={"P1"}, tracker=tracker
    )
    # The single line item is invalid, so the order should have no valid
    # items left and should be dropped entirely.
    assert len(items_clean) == 0
    assert len(orders_clean) == 0
