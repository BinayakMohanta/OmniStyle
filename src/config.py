"""
Centralized configuration for the OmniStyle Customer Intelligence Platform.

All environment variables are loaded here via python-dotenv so that every
other module in the project reads configuration from a single, consistent
place. No secrets are hardcoded anywhere in this project.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DASHBOARD_SAMPLE_DATA_DIR = PROJECT_ROOT / "dashboard" / "sample_data"

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, DASHBOARD_SAMPLE_DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load environment variables from .env (if present)
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    """
    Fetch an environment variable with a clear, actionable error message
    if a required variable is missing.
    """
    value = os.getenv(name, default)
    if required and not value:
        raise EnvironmentError(
            f"Missing required environment variable '{name}'.\n"
            f"Please copy '.env.example' to '.env' and set '{name}'.\n"
            f"See README.md 'Environment Variable Setup' section for details."
        )
    return value


# ---------------------------------------------------------------------------
# MongoDB configuration
# ---------------------------------------------------------------------------
MONGODB_URI = _get_env("MONGODB_URI", required=False)
MONGODB_DB = _get_env("MONGODB_DB", default="omnistyle_intelligence")

MONGO_COLLECTIONS = {
    "customers": "customers",
    "products": "products",
    "orders": "orders",
}

# ---------------------------------------------------------------------------
# MySQL configuration
# ---------------------------------------------------------------------------
MYSQL_HOST = _get_env("MYSQL_HOST", default="localhost")
MYSQL_PORT = int(_get_env("MYSQL_PORT", default="3306"))
MYSQL_USER = _get_env("MYSQL_USER", default="root")
MYSQL_PASSWORD = _get_env("MYSQL_PASSWORD", default="")
MYSQL_DATABASE = _get_env("MYSQL_DATABASE", default="omnistyle_analytics")


def get_mysql_uri(include_database: bool = True) -> str:
    """Build a SQLAlchemy-compatible MySQL connection URI (PyMySQL driver)."""
    base = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}"
    if include_database:
        return f"{base}/{MYSQL_DATABASE}"
    return base + "/"


# ---------------------------------------------------------------------------
# Reproducibility & synthetic data volume
# ---------------------------------------------------------------------------
RANDOM_SEED = int(_get_env("RANDOM_SEED", default="42"))
NUM_CUSTOMERS = int(_get_env("NUM_CUSTOMERS", default="5000"))
NUM_PRODUCTS = int(_get_env("NUM_PRODUCTS", default="200"))
NUM_ORDERS = int(_get_env("NUM_ORDERS", default="35000"))

# Reference "today" for the whole pipeline so that recency / churn windows
# are reproducible across runs regardless of when the code is executed.
REFERENCE_DATE_STR = "2026-07-01"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger configured with a consistent format.
    Safe to call multiple times (handlers are not duplicated).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
