# Data Directory

This folder is intentionally kept empty in version control (only `.gitkeep`
placeholders are committed). All CSV files here are **generated locally** by
running the pipeline — they are not checked into the repository because they
are synthetic and reproducible.

## `raw/`
Populated by `src/data_ingestion.py`. Contains a 1:1 CSV export of the
MongoDB collections:

- `customers.csv`
- `products.csv`
- `orders.csv` (order header rows)
- `order_items.csv` (exploded embedded `items` array, one row per line item)

## `processed/`
Populated by `src/data_cleaning.py` and `src/feature_engineering.py`.
Contains cleaned and feature-engineered datasets ready for MySQL loading,
SQL analysis, machine learning, and Power BI:

- `customers_clean.csv`
- `products_clean.csv`
- `orders_clean.csv`
- `order_items_clean.csv`
- `data_quality_report.csv`
- `customer_features.csv`
- `rfm_features.csv`
- `churn_dataset.csv`
- `customer_segments.csv` (after running segmentation)

To regenerate everything from scratch, follow the step-by-step instructions
in the main [README.md](../README.md).
