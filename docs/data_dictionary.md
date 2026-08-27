# Data Dictionary

All column names below were taken directly from the project's generated
CSV headers (not reconstructed from memory), so this document matches the
actual code exactly.

## 1. Raw data — `data/raw/` (MongoDB export / synthetic generator output)

### `customers.csv`

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Unique customer identifier, e.g. `CUST000001` |
| `name` | string | Customer full name (synthetic) |
| `email` | string | Synthetic email address |
| `gender` | string | `Female`, `Male`, or `Other` |
| `age` | int | Customer age |
| `city` | string | City |
| `state` | string | US state abbreviation |
| `region` | string | One of `North`, `South`, `East`, `West`, `Central` |
| `signup_date` | date (ISO) | Date the customer signed up |
| `loyalty_tier` | string | `Bronze`, `Silver`, `Gold`, `Platinum` |
| `loyalty_points` | int | Accumulated loyalty points |

### `products.csv`

| Column | Type | Description |
|---|---|---|
| `product_id` | string | Unique product identifier, e.g. `PROD00001` |
| `product_name` | string | Generated product name |
| `category` | string | `Men`, `Women`, `Kids`, `Accessories`, `Footwear` |
| `subcategory` | string | e.g. `Shirts`, `Dresses`, `Sneakers` |
| `brand` | string | One of the synthetic OmniStyle house brands |
| `unit_cost` | float | Cost to OmniStyle per unit ($) |
| `unit_price` | float | List price per unit ($) |

### `orders.csv` (order header)

| Column | Type | Description |
|---|---|---|
| `order_id` | string | Unique order identifier, e.g. `ORD0000001` |
| `customer_id` | string | FK to `customers.customer_id` |
| `order_date` | date (ISO) | Date the order was placed |
| `total_amount` | float | Order total ($), reconciled against line items during cleaning |
| `payment_method` | string | `Credit Card`, `Debit Card`, `UPI`, `Wallet`, `Cash on Delivery` |
| `channel` | string | `Online` or `In-Store` |
| `store_region` | string | Region the order was fulfilled from |

### `order_items.csv` (exploded line items)

| Column | Type | Description |
|---|---|---|
| `order_id` | string | FK to `orders.order_id` |
| `product_id` | string | FK to `products.product_id` |
| `quantity` | int | Units purchased in this line item |
| `unit_price` | float | Price per unit at time of sale ($) |
| `discount` | float | Discount fraction applied (0.0–0.9) |
| `line_total` | float | `unit_price * quantity * (1 - discount)` |

## 2. Cleaned data — `data/processed/*_clean.csv`

`customers_clean.csv`, `products_clean.csv`, `orders_clean.csv`, and
`order_items_clean.csv` share the **exact same columns** as their raw
counterparts above — cleaning fixes/validates values in place, it does not
add or rename columns. See `src/data_cleaning.py` for the exact validation
rules applied to each column.

### `data_quality_report.csv`

| Column | Type | Description |
|---|---|---|
| `dataset` | string | Which table the check applied to (`customers`, `products`, `orders`, `order_items`) |
| `check` | string | Name of the validation check, e.g. `duplicate_customer_id`, `invalid_age` |
| `issue_count` | int | Number of rows affected |
| `action_taken` | string | What the pipeline did about it (e.g. "dropped duplicates", "clipped to 0") |

## 3. RFM / feature data — `data/processed/`

### `rfm_features.csv`

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Customer identifier |
| `recency_days` | float | Days since the customer's last purchase as of the cutoff date (or since signup if no purchases) |
| `frequency` | int | Number of orders placed on/before the cutoff date |
| `monetary` | float | Total spend ($) on/before the cutoff date |
| `avg_order_value` | float | `monetary / frequency` (0 if frequency is 0) |
| `R_score` / `F_score` / `M_score` | int (1–5) | RFM quintile scores (5 = best) |
| `RFM_score` | string | Concatenation of R/F/M scores, e.g. `"453"` |

### `customer_features.csv`

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Customer identifier |
| `loyalty_tier`, `loyalty_points`, `region`, `age`, `gender` | — | Copied from `customers_clean.csv` |
| `customer_tenure_days` | int | Days between signup and the cutoff date |
| `recency_days`, `frequency`, `monetary`, `avg_order_value` | — | Same definitions as in `rfm_features.csv` |
| `total_items_purchased` | float | Sum of `quantity` across all line items on/before cutoff |
| `unique_products_purchased` | float | Distinct `product_id`s purchased |
| `unique_categories_purchased` | float | Distinct product categories purchased |
| `average_discount` | float | Mean discount fraction across the customer's line items |
| `purchase_trend_ratio` | float | Spend in the most recent 90 days of the window ÷ spend in the prior 90 days (>1 = accelerating, capped conceptually, defaults to 1.0 for customers with no prior-period spend and no recent spend) |
| `loyalty_tier_encoded` | int (0–3) | `Bronze=0, Silver=1, Gold=2, Platinum=3` |

### `churn_dataset.csv`

All columns from `customer_features.csv`, plus:

| Column | Type | Description |
|---|---|---|
| `churned` | int (0/1) | 1 if the customer made **zero** purchases in the 90-day observation window after the cutoff date, 0 otherwise. Only customers who signed up on/before the cutoff are included. See `docs/methodology.md` for the full definition. |

## 4. Model outputs

### `data/processed/customer_segments.csv`

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Customer identifier |
| `recency_days`, `frequency`, `monetary` | — | Same as `rfm_features.csv`, as of the cutoff date |
| `cluster` | int | Raw K-Means cluster ID (not business-meaningful on its own) |
| `segment` | string | Business label derived from the cluster's relative RFM ranking: `Champions`, `Loyal Customers`, `Potential Loyalists`, `New Customers`, `At Risk`, or `Hibernating` |

### `data/processed/churn_predictions.csv`

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Customer identifier (test-set customers only) |
| `actual_churned` | int (0/1) | Ground-truth label from `churn_dataset.csv` |
| `predicted_churned` | int (0/1) | Model's binary prediction |
| `churn_probability` | float (0–1) | Model's predicted probability of churn |

### `models/model_metrics.json`

| Key | Description |
|---|---|
| `best_model` | Name of the selected model (`"Random Forest"` or `"Logistic Regression"`) |
| `logistic_regression` / `random_forest` | Nested objects with `precision`, `recall`, `f1_score`, `roc_auc` on the held-out test set |
| `random_forest_cv_roc_auc_mean` / `_std` | 5-fold cross-validated ROC-AUC for the Random Forest |
| `churn_feature_columns` | Exact list of feature columns used to train the churn models |

### `reports/shap_feature_importance.csv`

| Column | Type | Description |
|---|---|---|
| `feature` | string | Feature name (matches `churn_feature_columns`) |
| `mean_abs_shap` | float | Mean absolute SHAP value across the test set (higher = more influential) |

## 5. MySQL analytics schema — `sql/schema.sql`

### `dim_customer`
`customer_id` (PK), `name`, `email`, `gender`, `age`, `city`, `state`,
`region`, `signup_date`, `loyalty_tier`, `loyalty_points` — same fields as
`customers_clean.csv`.

### `dim_product`
`product_id` (PK), `product_name`, `category`, `subcategory`, `brand`,
`unit_cost`, `unit_price` — same fields as `products_clean.csv`.

### `dim_date`
`date_key` (PK, DATE), `year`, `quarter`, `month`, `month_name`, `day`,
`day_of_week`, `day_name`, `is_weekend` — one row per calendar day spanning
the observed order date range (built by
`src/load_to_mysql.py::_build_dim_date`).

### `fact_sales`
`sales_id` (PK, auto-increment), `order_id`, `customer_id` (FK →
`dim_customer`), `product_id` (FK → `dim_product`), `date_key` (FK →
`dim_date`), `quantity`, `unit_price`, `discount`, `line_total`,
`payment_method`, `channel`, `store_region` — one row per order line item
(i.e., the same grain as `order_items_clean.csv`, joined with order header
context).

## 6. Power BI exports — `dashboard/sample_data/`

| File | Grain | Notes |
|---|---|---|
| `sales_detail.csv` | one row per order line item | Written by `src/train_model.py`; line items + order + product context |
| `customer_360.csv` | one row per customer | Written by `src/train_model.py`; customer + features + segment + churn probability (may contain duplicate `_feat`-suffixed columns from the merge — see `dashboard/README.md`) |
| `sales_dashboard.csv` | one row per order line item | Written by `src/export_dashboard_data.py`; cleaner column set + derived `order_year`/`order_month`/`gross_margin` |
| `customer_dashboard.csv` | one row per customer | Written by `src/export_dashboard_data.py`; customer + features + segment + `churn_probability` + derived `churn_risk_band` (`Low`/`Medium`/`High`) |
| `customer_segments.csv` | one row per customer | Direct copy of `data/processed/customer_segments.csv` |
| `churn_predictions.csv` | one row per scored (test-set) customer | Direct copy of `data/processed/churn_predictions.csv` |
| `dim_date.csv` | one row per calendar day | Same construction as the MySQL `dim_date` table |
