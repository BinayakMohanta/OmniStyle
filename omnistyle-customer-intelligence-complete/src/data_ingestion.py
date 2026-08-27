"""
src/data_ingestion.py
======================

Responsible for:

1. Generating deterministic, behaviorally-realistic synthetic retail data
   (customers, products, orders) for the fictional OmniStyle clothing brand.
2. Connecting to the user's EXISTING MongoDB Atlas cluster (via MONGODB_URI)
   and writing that data into a dedicated database (MONGODB_DB, default
   "omnistyle_intelligence") without touching any other database or
   collection on the cluster.
3. Exporting the MongoDB collections back out to flat CSV files in
   data/raw/ so the rest of the pipeline (which is MongoDB-agnostic) can
   run purely off CSVs.

If MONGODB_URI is not set, the script still works end-to-end: it generates
the synthetic data and writes directly to data/raw/ CSVs, skipping Mongo.
This keeps the project runnable even without Atlas credentials configured,
while still demonstrating the full MongoDB integration when credentials
are supplied.

Run as:
    python -m src.data_ingestion
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    MONGODB_DB,
    MONGODB_URI,
    MONGO_COLLECTIONS,
    NUM_CUSTOMERS,
    NUM_ORDERS,
    NUM_PRODUCTS,
    RANDOM_SEED,
    REFERENCE_DATE_STR,
    DATA_RAW_DIR,
    get_logger,
)

logger = get_logger(__name__)

REFERENCE_DATE = datetime.strptime(REFERENCE_DATE_STR, "%Y-%m-%d").date()

# ---------------------------------------------------------------------------
# Reference domain data
# ---------------------------------------------------------------------------
REGIONS = ["North", "South", "East", "West", "Central"]

REGION_STATE_CITY = {
    "North": [("New York", "NY"), ("Boston", "MA"), ("Buffalo", "NY")],
    "South": [("Houston", "TX"), ("Atlanta", "GA"), ("Miami", "FL")],
    "East": [("Philadelphia", "PA"), ("Newark", "NJ"), ("Baltimore", "MD")],
    "West": [("Los Angeles", "CA"), ("San Francisco", "CA"), ("Seattle", "WA")],
    "Central": [("Chicago", "IL"), ("Dallas", "TX"), ("Denver", "CO")],
}

LOYALTY_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
LOYALTY_TIER_WEIGHTS = [0.45, 0.30, 0.18, 0.07]

CATEGORY_SUBCATEGORIES = {
    "Men": ["T-Shirts", "Shirts", "Jeans", "Jackets", "Trousers"],
    "Women": ["Dresses", "Tops", "Jeans", "Jackets", "Skirts"],
    "Kids": ["T-Shirts", "Shorts", "Dresses", "Jackets"],
    "Accessories": ["Belts", "Bags", "Sunglasses", "Watches", "Hats"],
    "Footwear": ["Sneakers", "Formal Shoes", "Sandals", "Boots"],
}

BRANDS = [
    "OmniStyle Basics", "UrbanThread", "MetroFit", "NorthPeak",
    "AeroWear", "ClassicLane", "VelvetCo", "PulseActive",
]

PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Wallet", "Cash on Delivery"]
CHANNELS = ["Online", "In-Store"]

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Ananya", "Diya", "Isha", "Kabir", "Meera",
    "Rohan", "Sara", "James", "Emma", "Liam", "Olivia", "Noah", "Ava",
    "Sophia", "Ethan", "Mia", "Lucas", "Priya", "Arjun", "Neha", "Karan",
    "Zara", "Wei", "Mei", "Carlos", "Sofia", "Lucia",
]
LAST_NAMES = [
    "Sharma", "Verma", "Iyer", "Khan", "Patel", "Nair", "Smith", "Johnson",
    "Williams", "Brown", "Garcia", "Martinez", "Chen", "Wang", "Kumar",
    "Singh", "Reddy", "Das", "Gupta", "Mehta",
]

# Seasonal purchase multipliers by month (1=Jan ... 12=Dec).
# Clothing retail typically peaks around back-to-school (Aug/Sep),
# festive/holiday season (Nov/Dec), and summer sales (Jun).
SEASONAL_MULTIPLIER = {
    1: 0.85, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.10,
    7: 1.00, 8: 1.15, 9: 1.15, 10: 1.05, 11: 1.30, 12: 1.35,
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _random_date_between(rng: random.Random, start: date, end: date) -> date:
    delta_days = (end - start).days
    if delta_days <= 0:
        return start
    return start + timedelta(days=rng.randint(0, delta_days))


# ---------------------------------------------------------------------------
# Customer behavioral archetypes
# ---------------------------------------------------------------------------
# Each archetype drives correlated behavior across signup recency, order
# frequency, order recency, and discount sensitivity, so that downstream
# RFM / churn analysis reflects genuine behavioral patterns rather than
# independent random noise.
ARCHETYPES = [
    # name,              weight, order_rate_per_month, recency_bias_days, discount_affinity
    ("highly_active",     0.12, (3.5, 6.0), (0, 20),    (0.05, 0.15)),
    ("loyal",             0.18, (1.5, 3.0), (0, 45),    (0.05, 0.20)),
    ("new_customer",      0.15, (1.0, 2.5), (0, 60),    (0.10, 0.25)),
    ("at_risk",           0.20, (0.5, 1.2), (90, 180),  (0.15, 0.30)),
    ("inactive",          0.20, (0.1, 0.4), (150, 300), (0.10, 0.25)),
    ("churned",           0.15, (0.0, 0.1), (240, 420), (0.05, 0.20)),
]


def _assign_archetype(rng: random.Random) -> str:
    names = [a[0] for a in ARCHETYPES]
    weights = [a[1] for a in ARCHETYPES]
    return rng.choices(names, weights=weights, k=1)[0]


ARCHETYPE_LOOKUP = {a[0]: a for a in ARCHETYPES}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def generate_customers(n: int, rng: random.Random) -> list[dict[str, Any]]:
    logger.info("Generating %d synthetic customers...", n)
    customers = []
    earliest_signup = REFERENCE_DATE - timedelta(days=3 * 365)

    for i in range(1, n + 1):
        customer_id = f"CUST{i:06d}"
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}{i}@example.com"
        gender = rng.choices(["Female", "Male", "Other"], weights=[0.48, 0.48, 0.04])[0]
        age = int(np.clip(rng.gauss(34, 11), 18, 75))

        region = rng.choice(REGIONS)
        city, state = rng.choice(REGION_STATE_CITY[region])

        archetype = _assign_archetype(rng)
        _, _, _, recency_range, _ = ARCHETYPE_LOOKUP[archetype]

        # Loyal / highly_active customers tend to have longer tenure.
        if archetype in ("loyal", "highly_active"):
            signup_start = earliest_signup
            signup_end = REFERENCE_DATE - timedelta(days=200)
        elif archetype == "new_customer":
            signup_start = REFERENCE_DATE - timedelta(days=120)
            signup_end = REFERENCE_DATE - timedelta(days=1)
        else:
            signup_start = earliest_signup
            signup_end = REFERENCE_DATE - timedelta(days=30)

        signup_date = _random_date_between(rng, signup_start, signup_end)

        # Loyalty tier correlates loosely with tenure and archetype.
        tenure_days = (REFERENCE_DATE - signup_date).days
        if archetype in ("loyal", "highly_active") and tenure_days > 400:
            tier = rng.choices(LOYALTY_TIERS, weights=[0.15, 0.30, 0.35, 0.20])[0]
        elif archetype == "new_customer":
            tier = rng.choices(LOYALTY_TIERS, weights=[0.70, 0.25, 0.04, 0.01])[0]
        else:
            tier = rng.choices(LOYALTY_TIERS, weights=LOYALTY_TIER_WEIGHTS)[0]

        tier_points_base = {"Bronze": 150, "Silver": 600, "Gold": 1500, "Platinum": 4000}
        loyalty_points = int(max(0, rng.gauss(tier_points_base[tier], tier_points_base[tier] * 0.25)))

        customers.append(
            {
                "customer_id": customer_id,
                "name": name,
                "email": email,
                "gender": gender,
                "age": age,
                "city": city,
                "state": state,
                "region": region,
                "signup_date": signup_date.isoformat(),
                "loyalty_tier": tier,
                "loyalty_points": loyalty_points,
                # internal field used only for order generation; stripped before export
                "_archetype": archetype,
            }
        )

    logger.info("Customer generation complete.")
    return customers


def generate_products(n: int, rng: random.Random) -> list[dict[str, Any]]:
    logger.info("Generating %d synthetic products...", n)
    products = []
    categories = list(CATEGORY_SUBCATEGORIES.keys())

    for i in range(1, n + 1):
        product_id = f"PROD{i:05d}"
        category = rng.choice(categories)
        subcategory = rng.choice(CATEGORY_SUBCATEGORIES[category])
        brand = rng.choice(BRANDS)
        product_name = f"{brand} {subcategory} {rng.choice(['Classic', 'Essential', 'Premium', 'Everyday', 'Pro'])}"

        # Category-dependent cost/price ranges for realism.
        price_ranges = {
            "Men": (15, 90), "Women": (15, 110), "Kids": (10, 60),
            "Accessories": (8, 120), "Footwear": (25, 160),
        }
        low, high = price_ranges[category]
        unit_price = round(rng.uniform(low, high), 2)
        margin = rng.uniform(0.35, 0.55)  # cost as a fraction of price
        unit_cost = round(unit_price * (1 - margin), 2)

        products.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "subcategory": subcategory,
                "brand": brand,
                "unit_cost": unit_cost,
                "unit_price": unit_price,
            }
        )

    logger.info("Product generation complete.")
    return products


# Region -> category affinity (soft preference weights), to create
# realistic region/product relationships.
REGION_CATEGORY_AFFINITY = {
    "North": {"Men": 1.0, "Women": 1.1, "Kids": 0.9, "Accessories": 1.0, "Footwear": 1.1},
    "South": {"Men": 1.1, "Women": 0.9, "Kids": 1.0, "Accessories": 0.9, "Footwear": 1.0},
    "East":  {"Men": 1.0, "Women": 1.0, "Kids": 1.0, "Accessories": 1.1, "Footwear": 0.9},
    "West":  {"Men": 0.9, "Women": 1.2, "Kids": 0.9, "Accessories": 1.1, "Footwear": 1.1},
    "Central": {"Men": 1.0, "Women": 1.0, "Kids": 1.1, "Accessories": 1.0, "Footwear": 1.0},
}


def _weighted_products_for_region(products: pd.DataFrame, region: str, k: int, rng: random.Random) -> pd.DataFrame:
    affinity = REGION_CATEGORY_AFFINITY[region]
    weights = products["category"].map(affinity).fillna(1.0).to_numpy()
    weights = weights / weights.sum()
    idx = rng.choices(range(len(products)), weights=list(weights), k=k)
    return products.iloc[idx]


def generate_orders(
    customers: list[dict[str, Any]],
    products: list[dict[str, Any]],
    target_orders: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    logger.info("Generating approximately %d synthetic orders...", target_orders)

    products_df = pd.DataFrame(products)
    orders: list[dict[str, Any]] = []
    order_counter = 1

    for cust in customers:
        archetype = cust["_archetype"]
        _, _, rate_range, recency_range, discount_range = ARCHETYPE_LOOKUP[archetype]

        signup_date = datetime.strptime(cust["signup_date"], "%Y-%m-%d").date()
        tenure_months = max(1, (REFERENCE_DATE - signup_date).days // 30)

        monthly_rate = rng.uniform(*rate_range)
        # Scale down globally so total orders land near target_orders.
        expected_orders = monthly_rate * tenure_months

        # Determine the customer's "last active" recency window based on archetype.
        recency_low, recency_high = recency_range
        last_active_days_ago = rng.randint(recency_low, recency_high)
        last_active_date = REFERENCE_DATE - timedelta(days=last_active_days_ago)
        last_active_date = max(last_active_date, signup_date + timedelta(days=1))

        n_orders_for_customer = int(np.random.poisson(max(expected_orders * 0.22, 0.05)))
        n_orders_for_customer = min(n_orders_for_customer, 60)

        if n_orders_for_customer == 0:
            continue

        region = cust["region"]

        for _ in range(n_orders_for_customer):
            # Spread order dates between signup and the customer's last-active date,
            # weighted toward more recent dates for active archetypes.
            order_date = _random_date_between(rng, signup_date + timedelta(days=1), last_active_date)

            # Apply seasonality by re-rolling with a seasonal acceptance check.
            seasonal_weight = SEASONAL_MULTIPLIER[order_date.month]
            if rng.random() > min(seasonal_weight / 1.35, 1.0):
                # small chance to skip / redraw to bias toward high season
                alt_date = _random_date_between(rng, signup_date + timedelta(days=1), last_active_date)
                if SEASONAL_MULTIPLIER[alt_date.month] > seasonal_weight:
                    order_date = alt_date

            channel = rng.choices(CHANNELS, weights=[0.68, 0.32])[0]
            payment_method = rng.choice(PAYMENT_METHODS)

            n_items = rng.choices([1, 2, 3, 4], weights=[0.45, 0.30, 0.15, 0.10])[0]
            item_products = _weighted_products_for_region(products_df, region, n_items, rng)

            discount_low, discount_high = discount_range
            items = []
            order_total = 0.0
            for _, prod in item_products.iterrows():
                quantity = rng.choices([1, 2, 3], weights=[0.7, 0.22, 0.08])[0]
                discount = round(rng.uniform(discount_low, discount_high), 2)
                unit_price = float(prod["unit_price"])
                line_total = round(unit_price * quantity * (1 - discount), 2)
                order_total += line_total
                items.append(
                    {
                        "product_id": prod["product_id"],
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "discount": discount,
                        "line_total": line_total,
                    }
                )

            order_id = f"ORD{order_counter:07d}"
            order_counter += 1

            orders.append(
                {
                    "order_id": order_id,
                    "customer_id": cust["customer_id"],
                    "order_date": order_date.isoformat(),
                    "items": items,
                    "total_amount": round(order_total, 2),
                    "payment_method": payment_method,
                    "channel": channel,
                    "store_region": region,
                }
            )

    logger.info("Generated %d total orders (target was %d).", len(orders), target_orders)
    return orders


def strip_internal_fields(customers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return customer dicts without the internal '_archetype' helper field."""
    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in customers]


def generate_all_synthetic_data(
    num_customers: int = NUM_CUSTOMERS,
    num_products: int = NUM_PRODUCTS,
    num_orders: int = NUM_ORDERS,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Generate the full synthetic dataset deterministically."""
    _seed_everything(seed)
    rng = random.Random(seed)

    customers = generate_customers(num_customers, rng)
    products = generate_products(num_products, rng)
    orders = generate_orders(customers, products, num_orders, rng)

    return customers, products, orders


# ---------------------------------------------------------------------------
# MongoDB integration
# ---------------------------------------------------------------------------
def get_mongo_database():
    """
    Connect to the user's existing MongoDB Atlas cluster and return a handle
    to the dedicated OmniStyle database. Returns None if MONGODB_URI is not
    configured, allowing the pipeline to fall back to CSV-only mode.
    """
    if not MONGODB_URI:
        logger.warning(
            "MONGODB_URI is not set. Skipping MongoDB and writing synthetic "
            "data directly to CSV files in data/raw/. Set MONGODB_URI in "
            "your .env file to enable full MongoDB integration."
        )
        return None

    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError
    except ImportError as exc:
        raise ImportError("pymongo is required. Install with: pip install pymongo") from exc

    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not connect to MongoDB Atlas: %s", exc)
        logger.error(
            "Falling back to CSV-only mode. Check that MONGODB_URI is correct "
            "and that your current IP is allow-listed in Atlas Network Access."
        )
        return None

    db = client[MONGODB_DB]
    logger.info("Connected to MongoDB Atlas. Using database '%s' (existing cluster, "
                "no new cluster created).", MONGODB_DB)
    return db


def ensure_indexes(db) -> None:
    """Create sensible indexes on the OmniStyle collections (idempotent)."""
    db[MONGO_COLLECTIONS["customers"]].create_index("customer_id", unique=True)
    db[MONGO_COLLECTIONS["products"]].create_index("product_id", unique=True)
    db[MONGO_COLLECTIONS["orders"]].create_index("order_id", unique=True)
    db[MONGO_COLLECTIONS["orders"]].create_index("customer_id")
    db[MONGO_COLLECTIONS["orders"]].create_index("order_date")
    logger.info("Ensured indexes on customers, products, and orders collections.")


def upsert_documents(collection, documents: list[dict], key_field: str) -> None:
    """Insert documents while avoiding duplicates, keyed on key_field."""
    from pymongo import UpdateOne

    if not documents:
        return

    operations = [
        UpdateOne({key_field: doc[key_field]}, {"$set": doc}, upsert=True)
        for doc in documents
    ]
    # Bulk write in batches to avoid oversized requests.
    batch_size = 2000
    total_upserted = 0
    for i in range(0, len(operations), batch_size):
        batch = operations[i : i + batch_size]
        result = collection.bulk_write(batch, ordered=False)
        total_upserted += (result.upserted_count or 0) + (result.modified_count or 0)
    logger.info("Upserted %d documents into '%s'.", len(documents), collection.name)


def load_into_mongo(db, customers: list[dict], products: list[dict], orders: list[dict]) -> None:
    ensure_indexes(db)
    upsert_documents(db[MONGO_COLLECTIONS["customers"]], strip_internal_fields(customers), "customer_id")
    upsert_documents(db[MONGO_COLLECTIONS["products"]], products, "product_id")
    upsert_documents(db[MONGO_COLLECTIONS["orders"]], orders, "order_id")


def export_mongo_to_csv(db) -> None:
    """Export MongoDB collections to CSVs in data/raw/."""
    customers = list(db[MONGO_COLLECTIONS["customers"]].find({}, {"_id": 0}))
    products = list(db[MONGO_COLLECTIONS["products"]].find({}, {"_id": 0}))
    orders = list(db[MONGO_COLLECTIONS["orders"]].find({}, {"_id": 0}))
    _write_csvs(customers, products, orders)


def _write_csvs(customers: list[dict], products: list[dict], orders: list[dict]) -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    customers_df = pd.DataFrame(strip_internal_fields(customers))
    products_df = pd.DataFrame(products)

    order_rows = []
    item_rows = []
    for order in orders:
        order_rows.append(
            {
                "order_id": order["order_id"],
                "customer_id": order["customer_id"],
                "order_date": order["order_date"],
                "total_amount": order["total_amount"],
                "payment_method": order["payment_method"],
                "channel": order["channel"],
                "store_region": order["store_region"],
            }
        )
        for item in order["items"]:
            item_rows.append(
                {
                    "order_id": order["order_id"],
                    "product_id": item["product_id"],
                    "quantity": item["quantity"],
                    "unit_price": item["unit_price"],
                    "discount": item["discount"],
                    "line_total": item["line_total"],
                }
            )

    orders_df = pd.DataFrame(order_rows)
    order_items_df = pd.DataFrame(item_rows)

    customers_df.to_csv(DATA_RAW_DIR / "customers.csv", index=False)
    products_df.to_csv(DATA_RAW_DIR / "products.csv", index=False)
    orders_df.to_csv(DATA_RAW_DIR / "orders.csv", index=False)
    order_items_df.to_csv(DATA_RAW_DIR / "order_items.csv", index=False)

    logger.info(
        "Wrote raw CSVs to %s: customers=%d, products=%d, orders=%d, order_items=%d",
        DATA_RAW_DIR, len(customers_df), len(products_df), len(orders_df), len(order_items_df),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="OmniStyle data ingestion pipeline")
    parser.add_argument("--force-regenerate", action="store_true",
                         help="Regenerate synthetic data even if collections already contain data.")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("OmniStyle Customer Intelligence Platform - Data Ingestion")
    logger.info("=" * 70)

    db = get_mongo_database()

    if db is not None:
        counts = {
            name: db[coll].count_documents({})
            for name, coll in MONGO_COLLECTIONS.items()
        }
        logger.info("Existing document counts: %s", counts)

        needs_generation = args.force_regenerate or any(c == 0 for c in counts.values())

        if needs_generation:
            logger.info("Collections are empty (or --force-regenerate set). Generating synthetic data...")
            customers, products, orders = generate_all_synthetic_data()
            load_into_mongo(db, customers, products, orders)
        else:
            logger.info("Collections already contain data. Skipping generation (use --force-regenerate to override).")

        logger.info("Exporting MongoDB collections to CSV files...")
        export_mongo_to_csv(db)
    else:
        logger.info("Running in CSV-only mode (no MongoDB connection).")
        customers, products, orders = generate_all_synthetic_data()
        _write_csvs(customers, products, orders)

    logger.info("Data ingestion complete. Raw CSVs are available in data/raw/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
