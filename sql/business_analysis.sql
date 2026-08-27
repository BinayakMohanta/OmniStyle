-- =========================================================================
-- OmniStyle Customer Intelligence Platform
-- Business Analysis Queries
-- =========================================================================
-- Run against the omnistyle_analytics database after loading data with
-- src/load_to_mysql.py (see README.md for the exact command).
--
--   mysql -u <user> -p omnistyle_analytics < sql/business_analysis.sql
-- =========================================================================

USE omnistyle_analytics;

-- -------------------------------------------------------------------------
-- 1. TOTAL REVENUE
-- Business question: What is our total revenue across all recorded sales?
-- -------------------------------------------------------------------------
SELECT
    ROUND(SUM(line_total), 2) AS total_revenue,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_id) AS total_customers
FROM fact_sales;

-- -------------------------------------------------------------------------
-- 2. MONTHLY REVENUE TREND
-- Business question: How has revenue trended month over month?
-- -------------------------------------------------------------------------
SELECT
    d.year,
    d.month,
    d.month_name,
    ROUND(SUM(f.line_total), 2) AS monthly_revenue,
    COUNT(DISTINCT f.order_id) AS monthly_orders
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;

-- -------------------------------------------------------------------------
-- 3. MONTH-OVER-MONTH GROWTH
-- Business question: What is the % change in revenue vs. the prior month?
-- -------------------------------------------------------------------------
WITH monthly_revenue AS (
    SELECT
        d.year,
        d.month,
        SUM(f.line_total) AS revenue
    FROM fact_sales f
    JOIN dim_date d ON f.date_key = d.date_key
    GROUP BY d.year, d.month
)
SELECT
    year,
    month,
    ROUND(revenue, 2) AS revenue,
    ROUND(LAG(revenue) OVER (ORDER BY year, month), 2) AS prev_month_revenue,
    ROUND(
        100.0 * (revenue - LAG(revenue) OVER (ORDER BY year, month))
        / NULLIF(LAG(revenue) OVER (ORDER BY year, month), 0), 2
    ) AS mom_growth_pct
FROM monthly_revenue
ORDER BY year, month;

-- -------------------------------------------------------------------------
-- 4. TOP PRODUCTS BY REVENUE
-- Business question: Which products generate the most revenue?
-- -------------------------------------------------------------------------
SELECT
    p.product_id,
    p.product_name,
    p.category,
    ROUND(SUM(f.line_total), 2) AS revenue,
    SUM(f.quantity) AS units_sold
FROM fact_sales f
JOIN dim_product p ON f.product_id = p.product_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY revenue DESC
LIMIT 20;

-- -------------------------------------------------------------------------
-- 5. BOTTOM-PERFORMING PRODUCTS
-- Business question: Which products are underperforming and worth reviewing?
-- -------------------------------------------------------------------------
SELECT
    p.product_id,
    p.product_name,
    p.category,
    ROUND(SUM(f.line_total), 2) AS revenue,
    SUM(f.quantity) AS units_sold
FROM fact_sales f
JOIN dim_product p ON f.product_id = p.product_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY revenue ASC
LIMIT 20;

-- -------------------------------------------------------------------------
-- 6. CATEGORY PERFORMANCE
-- Business question: How does each clothing category contribute to revenue?
-- -------------------------------------------------------------------------
SELECT
    p.category,
    ROUND(SUM(f.line_total), 2) AS revenue,
    SUM(f.quantity) AS units_sold,
    ROUND(SUM(f.line_total) * 100.0 / SUM(SUM(f.line_total)) OVER (), 2) AS pct_of_total_revenue
FROM fact_sales f
JOIN dim_product p ON f.product_id = p.product_id
GROUP BY p.category
ORDER BY revenue DESC;

-- -------------------------------------------------------------------------
-- 7. REGIONAL PERFORMANCE
-- Business question: Which regions drive the most sales and customers?
-- -------------------------------------------------------------------------
SELECT
    f.store_region,
    ROUND(SUM(f.line_total), 2) AS revenue,
    COUNT(DISTINCT f.order_id) AS orders,
    COUNT(DISTINCT f.customer_id) AS customers,
    ROUND(SUM(f.line_total) / COUNT(DISTINCT f.order_id), 2) AS avg_order_value
FROM fact_sales f
GROUP BY f.store_region
ORDER BY revenue DESC;

-- -------------------------------------------------------------------------
-- 8. CUSTOMER LIFETIME VALUE RANKING
-- Business question: Who are our highest lifetime-value customers?
-- -------------------------------------------------------------------------
SELECT
    c.customer_id,
    c.name,
    c.loyalty_tier,
    ROUND(SUM(f.line_total), 2) AS lifetime_value,
    COUNT(DISTINCT f.order_id) AS total_orders
FROM fact_sales f
JOIN dim_customer c ON f.customer_id = c.customer_id
GROUP BY c.customer_id, c.name, c.loyalty_tier
ORDER BY lifetime_value DESC
LIMIT 25;

-- -------------------------------------------------------------------------
-- 9. REPEAT CUSTOMER RATE
-- Business question: What share of customers have placed more than one order?
-- -------------------------------------------------------------------------
WITH order_counts AS (
    SELECT customer_id, COUNT(DISTINCT order_id) AS n_orders
    FROM fact_sales
    GROUP BY customer_id
)
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END) AS repeat_customers,
    ROUND(100.0 * SUM(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS repeat_customer_rate_pct
FROM order_counts;

-- -------------------------------------------------------------------------
-- 10. AVERAGE ORDER VALUE (overall and by channel)
-- Business question: What is the average order value, and does it vary by channel?
-- -------------------------------------------------------------------------
SELECT
    f.channel,
    ROUND(SUM(f.line_total) / COUNT(DISTINCT f.order_id), 2) AS avg_order_value,
    COUNT(DISTINCT f.order_id) AS orders
FROM fact_sales f
GROUP BY f.channel
ORDER BY avg_order_value DESC;

-- -------------------------------------------------------------------------
-- 11. LOYALTY TIER PERFORMANCE
-- Business question: How does spend differ across loyalty tiers?
-- -------------------------------------------------------------------------
SELECT
    c.loyalty_tier,
    COUNT(DISTINCT c.customer_id) AS customers,
    ROUND(SUM(f.line_total), 2) AS revenue,
    ROUND(SUM(f.line_total) / COUNT(DISTINCT c.customer_id), 2) AS revenue_per_customer
FROM dim_customer c
LEFT JOIN fact_sales f ON c.customer_id = f.customer_id
GROUP BY c.loyalty_tier
ORDER BY revenue_per_customer DESC;

-- -------------------------------------------------------------------------
-- 12. CUSTOMER INACTIVITY ANALYSIS
-- Business question: How many days has it been since each customer's last purchase,
-- and how many customers fall into each inactivity bucket?
-- -------------------------------------------------------------------------
WITH last_purchase AS (
    SELECT
        customer_id,
        MAX(date_key) AS last_purchase_date,
        DATEDIFF((SELECT MAX(date_key) FROM fact_sales), MAX(date_key)) AS days_since_last_purchase
    FROM fact_sales
    GROUP BY customer_id
)
SELECT
    CASE
        WHEN days_since_last_purchase <= 30 THEN '0-30 days'
        WHEN days_since_last_purchase <= 90 THEN '31-90 days'
        WHEN days_since_last_purchase <= 180 THEN '91-180 days'
        ELSE '180+ days'
    END AS inactivity_bucket,
    COUNT(*) AS customers
FROM last_purchase
GROUP BY inactivity_bucket
ORDER BY FIELD(inactivity_bucket, '0-30 days', '31-90 days', '91-180 days', '180+ days');

-- -------------------------------------------------------------------------
-- 13. RUNNING REVENUE TOTALS (cumulative revenue by month)
-- Business question: What does our cumulative revenue trajectory look like?
-- -------------------------------------------------------------------------
WITH monthly_revenue AS (
    SELECT d.year, d.month, SUM(f.line_total) AS revenue
    FROM fact_sales f
    JOIN dim_date d ON f.date_key = d.date_key
    GROUP BY d.year, d.month
)
SELECT
    year,
    month,
    ROUND(revenue, 2) AS monthly_revenue,
    ROUND(SUM(revenue) OVER (ORDER BY year, month), 2) AS cumulative_revenue
FROM monthly_revenue
ORDER BY year, month;

-- -------------------------------------------------------------------------
-- 14. TOP CUSTOMERS PER REGION (window function ranking)
-- Business question: Who is the top customer by revenue within each region?
-- -------------------------------------------------------------------------
WITH customer_region_revenue AS (
    SELECT
        c.region,
        c.customer_id,
        c.name,
        SUM(f.line_total) AS revenue,
        RANK() OVER (PARTITION BY c.region ORDER BY SUM(f.line_total) DESC) AS region_rank
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_id = c.customer_id
    GROUP BY c.region, c.customer_id, c.name
)
SELECT region, customer_id, name, ROUND(revenue, 2) AS revenue, region_rank
FROM customer_region_revenue
WHERE region_rank <= 5
ORDER BY region, region_rank;

-- -------------------------------------------------------------------------
-- 15. CUSTOMER PURCHASE FREQUENCY DISTRIBUTION
-- Business question: How many orders do customers typically place?
-- -------------------------------------------------------------------------
WITH order_counts AS (
    SELECT customer_id, COUNT(DISTINCT order_id) AS n_orders
    FROM fact_sales
    GROUP BY customer_id
)
SELECT
    CASE
        WHEN n_orders = 1 THEN '1 order'
        WHEN n_orders BETWEEN 2 AND 3 THEN '2-3 orders'
        WHEN n_orders BETWEEN 4 AND 6 THEN '4-6 orders'
        ELSE '7+ orders'
    END AS frequency_bucket,
    COUNT(*) AS customers
FROM order_counts
GROUP BY frequency_bucket
ORDER BY FIELD(frequency_bucket, '1 order', '2-3 orders', '4-6 orders', '7+ orders');

-- -------------------------------------------------------------------------
-- 16. PAYMENT METHOD PREFERENCE (bonus query)
-- Business question: Which payment methods are most popular, and do they
-- correlate with higher order values?
-- -------------------------------------------------------------------------
SELECT
    payment_method,
    COUNT(DISTINCT order_id) AS orders,
    ROUND(SUM(line_total) / COUNT(DISTINCT order_id), 2) AS avg_order_value
FROM fact_sales
GROUP BY payment_method
ORDER BY orders DESC;

-- -------------------------------------------------------------------------
-- 17. DISCOUNT IMPACT ON PURCHASING (bonus query)
-- Business question: Do higher discounts correlate with higher unit sales?
-- -------------------------------------------------------------------------
SELECT
    CASE
        WHEN discount = 0 THEN 'No discount'
        WHEN discount <= 0.10 THEN '1-10%'
        WHEN discount <= 0.20 THEN '11-20%'
        ELSE '20%+'
    END AS discount_band,
    COUNT(*) AS line_items,
    SUM(quantity) AS units_sold,
    ROUND(AVG(quantity), 2) AS avg_qty_per_line
FROM fact_sales
GROUP BY discount_band
ORDER BY FIELD(discount_band, 'No discount', '1-10%', '11-20%', '20%+');
