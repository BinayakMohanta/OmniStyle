# OmniStyle Power BI Dashboard Guide

This guide walks through building the OmniStyle Customer Intelligence
dashboard in Power BI Desktop using the CSVs in `dashboard/sample_data/`.

---

## 1. Data Model

### Import the tables

In Power BI Desktop: **Get Data → Text/CSV**, and import each file from
`dashboard/sample_data/`:

| File | Power BI table name | Role |
|---|---|---|
| `customers.csv` | `Customers` | Dimension |
| `products.csv` | `Products` | Dimension |
| `sales_detail.csv` | `SalesDetail` | Fact table (line-item grain) |
| `customer_360.csv` | `CustomerIntelligence` | Customer-level analytics (segment, churn) |

`orders.csv` is optional — `sales_detail.csv` already includes order-level
context (`order_date`, `channel`, `store_region`, `payment_method`) joined
onto each line item, so most visuals can be built directly from it.

### Recommended relationships

Build these relationships in the Power BI **Model** view:

- `SalesDetail[customer_id]` → `Customers[customer_id]` (many-to-one)
- `SalesDetail[product_id]` → `Products[product_id]` (many-to-one)
- `Customers[customer_id]` → `CustomerIntelligence[customer_id]` (one-to-one)

### Add a Date table

Power BI needs a proper date dimension for time-intelligence DAX functions.
In Power BI Desktop: **Modeling → New Table**, then:

```
DateTable =
CALENDAR (
    MIN ( SalesDetail[order_date] ),
    MAX ( SalesDetail[order_date] )
)
```

Mark it as a **Date Table** (Modeling → Mark as Date Table), then relate
`DateTable[Date]` → `SalesDetail[order_date]` (one-to-many).

---

## 2. Dashboard Pages

### Page 1: Executive Overview

**KPIs (as cards across the top):**
- Total Revenue
- Total Customers
- Total Orders
- Average Order Value
- Repeat Customer Rate

**Suggested visuals:**
- Line chart: Revenue trend by month (`DateTable[Month]` on axis, `[Total Revenue]` measure as value)
- KPI cards for the five metrics above
- Bar chart: Revenue by region

### Page 2: Sales Intelligence

**Suggested visuals:**
- Line chart: Revenue trend with a trend line (uses `[MoM Revenue Growth %]`)
- Bar chart: Category performance (`Products[category]` vs. `[Total Revenue]`)
- Table or bar chart: Top 10 products by revenue
- Map or bar chart: Regional performance (`SalesDetail[store_region]`)
- Combo chart: Revenue vs. order count by month, to visualize growth quality

### Page 3: Customer Intelligence

**Suggested visuals:**
- Donut chart: Customer segment distribution (`CustomerIntelligence[segment]`)
- Bar chart: Loyalty tier analysis (`CustomerIntelligence[loyalty_tier]` vs. average monetary value)
- Table: At-risk customers — filter `CustomerIntelligence[churn_probability] >= 0.6`,
  sorted descending, showing `customer_id`, `name`, `segment`, `churn_probability`
- Gauge or histogram: Distribution of `churn_probability` across all customers
- Card/table: Recommended actions per segment (static text box referencing
  `reports/business_recommendations.md`, or a table visual driven by a
  manually-entered "recommended action" column per segment)

---

## 3. Recommended DAX Measures

Paste these into a new measure table (e.g., create a blank table called
`_Measures` to keep them organized) or add them directly to `SalesDetail` /
`CustomerIntelligence`.

```DAX
Total Revenue = SUM ( SalesDetail[line_total] )

Total Orders = DISTINCTCOUNT ( SalesDetail[order_id] )

Total Customers = DISTINCTCOUNT ( SalesDetail[customer_id] )

Average Order Value =
DIVIDE ( [Total Revenue], [Total Orders], 0 )

Repeat Customer Rate =
VAR OrdersPerCustomer =
    SUMMARIZE (
        SalesDetail,
        SalesDetail[customer_id],
        "OrderCount", DISTINCTCOUNT ( SalesDetail[order_id] )
    )
VAR RepeatCustomers =
    COUNTROWS ( FILTER ( OrdersPerCustomer, [OrderCount] > 1 ) )
VAR TotalCustomers = COUNTROWS ( OrdersPerCustomer )
RETURN
    DIVIDE ( RepeatCustomers, TotalCustomers, 0 )

MoM Revenue Growth % =
VAR CurrentRevenue = [Total Revenue]
VAR PriorRevenue =
    CALCULATE ( [Total Revenue], DATEADD ( DateTable[Date], -1, MONTH ) )
RETURN
    DIVIDE ( CurrentRevenue - PriorRevenue, PriorRevenue, BLANK() )

Average Discount =
AVERAGE ( SalesDetail[discount] )

High Risk Customers =
CALCULATE (
    DISTINCTCOUNT ( CustomerIntelligence[customer_id] ),
    CustomerIntelligence[churn_probability] >= 0.6
)

Average Churn Probability =
AVERAGE ( CustomerIntelligence[churn_probability] )

Revenue per Customer =
DIVIDE ( [Total Revenue], [Total Customers], 0 )
```

---

## 4. Suggested Filters / Slicers

Add slicers for:
- `DateTable[Year]` / `DateTable[Month]`
- `Customers[region]`
- `Customers[loyalty_tier]`
- `CustomerIntelligence[segment]`
- `SalesDetail[channel]`

These let a business user drill from the Executive Overview down into
specific regions, tiers, or segments without switching pages.

---

## 5. Data Refresh

Because `sample_data/` is a static CSV export, refreshing the dashboard with
new data means re-running the pipeline (`python -m src.data_ingestion` →
`data_cleaning` → `feature_engineering` → `train_model`) and then clicking
**Refresh** in Power BI Desktop, since the CSV file paths stay the same.
