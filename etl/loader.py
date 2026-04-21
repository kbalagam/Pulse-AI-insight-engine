"""
Loads raw CSV files from the data/raw directory.

Each function returns a DataFrame with parsed dates and correct dtypes.
No business logic here — loading only.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "transactions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["product_id"] = df["product_id"].astype("Int64")
    return df


def load_customers() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "customers.csv")
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["signup_date_only"] = df["signup_date"].dt.date
    return df


def load_events() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "events.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["product_id"] = df["product_id"].astype("Int64")
    return df


def load_campaigns() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "campaigns.csv")
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def load_products() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "products.csv")
    df["launch_date"] = pd.to_datetime(df["launch_date"])
    df["product_label"] = df["brand"] + " - " + df["category"]
    return df


def load_all() -> dict:
    """Convenience loader that returns all tables as a named dict."""
    return {
        "transactions": load_transactions(),
        "customers": load_customers(),
        "events": load_events(),
        "campaigns": load_campaigns(),
        "products": load_products(),
    }
