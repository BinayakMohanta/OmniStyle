-- =========================================================================
-- OmniStyle Customer Intelligence Platform
-- MySQL Analytics Database Schema (Star Schema)
-- =========================================================================
-- This schema is intentionally simple (one fact table, three dimensions)
-- so it is easy to load, easy to query, and easy to model in Power BI.
--
-- Usage:
--   mysql -u <user> -p < sql/schema.sql
-- =========================================================================

CREATE DATABASE IF NOT EXISTS omnistyle_analytics
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE omnistyle_analytics;

-- -------------------------------------------------------------------------
-- Drop tables if re-running (fact first, due to FK constraints)
-- -------------------------------------------------------------------------
DROP TABLE IF EXISTS fact_sales;
DROP TABLE IF EXISTS dim_customer;
DROP TABLE IF EXISTS dim_product;
DROP TABLE IF EXISTS dim_date;

-- -------------------------------------------------------------------------
-- dim_customer: one row per customer
-- -------------------------------------------------------------------------
CREATE TABLE dim_customer (
    customer_id         VARCHAR(20)     NOT NULL,
    name                VARCHAR(150)    NOT NULL,
    email               VARCHAR(150)    NOT NULL,
    gender              VARCHAR(20)     NOT NULL,
    age                 INT             NOT NULL,
    city                VARCHAR(100)    NOT NULL,
    state               VARCHAR(50)     NOT NULL,
    region              VARCHAR(50)     NOT NULL,
    signup_date         DATE            NOT NULL,
    loyalty_tier        VARCHAR(20)     NOT NULL,
    loyalty_points      INT             NOT NULL DEFAULT 0,
    PRIMARY KEY (customer_id),
    INDEX idx_dim_customer_region (region),
    INDEX idx_dim_customer_loyalty_tier (loyalty_tier)
) ENGINE=InnoDB;

-- -------------------------------------------------------------------------
-- dim_product: one row per product (SCD Type 1 - simple overwrite)
-- -------------------------------------------------------------------------
CREATE TABLE dim_product (
    product_id          VARCHAR(20)     NOT NULL,
    product_name        VARCHAR(200)    NOT NULL,
    category            VARCHAR(50)     NOT NULL,
    subcategory         VARCHAR(50)     NOT NULL,
    brand               VARCHAR(100)    NOT NULL,
    unit_cost           DECIMAL(10, 2)  NOT NULL,
    unit_price          DECIMAL(10, 2)  NOT NULL,
    PRIMARY KEY (product_id),
    INDEX idx_dim_product_category (category),
    INDEX idx_dim_product_brand (brand)
) ENGINE=InnoDB;

-- -------------------------------------------------------------------------
-- dim_date: one row per calendar date, pre-populated by loader script
-- -------------------------------------------------------------------------
CREATE TABLE dim_date (
    date_key            DATE            NOT NULL,
    year                INT             NOT NULL,
    quarter             INT             NOT NULL,
    month               INT             NOT NULL,
    month_name          VARCHAR(20)     NOT NULL,
    day                 INT             NOT NULL,
    day_of_week         INT             NOT NULL,
    day_name            VARCHAR(20)     NOT NULL,
    is_weekend          TINYINT(1)      NOT NULL,
    PRIMARY KEY (date_key),
    INDEX idx_dim_date_year_month (year, month)
) ENGINE=InnoDB;

-- -------------------------------------------------------------------------
-- fact_sales: one row per order line item
-- -------------------------------------------------------------------------
CREATE TABLE fact_sales (
    sales_id            BIGINT AUTO_INCREMENT,
    order_id            VARCHAR(20)     NOT NULL,
    customer_id         VARCHAR(20)     NOT NULL,
    product_id          VARCHAR(20)     NOT NULL,
    date_key            DATE            NOT NULL,
    quantity            INT             NOT NULL,
    unit_price          DECIMAL(10, 2)  NOT NULL,
    discount            DECIMAL(5, 4)   NOT NULL,
    line_total          DECIMAL(12, 2)  NOT NULL,
    payment_method      VARCHAR(30)     NOT NULL,
    channel             VARCHAR(20)     NOT NULL,
    store_region        VARCHAR(50)     NOT NULL,
    PRIMARY KEY (sales_id),
    INDEX idx_fact_sales_customer (customer_id),
    INDEX idx_fact_sales_product (product_id),
    INDEX idx_fact_sales_date (date_key),
    INDEX idx_fact_sales_order (order_id),
    CONSTRAINT fk_fact_sales_customer FOREIGN KEY (customer_id)
        REFERENCES dim_customer (customer_id),
    CONSTRAINT fk_fact_sales_product FOREIGN KEY (product_id)
        REFERENCES dim_product (product_id),
    CONSTRAINT fk_fact_sales_date FOREIGN KEY (date_key)
        REFERENCES dim_date (date_key)
) ENGINE=InnoDB;
