"""
Cleans and standardizes raw DataFrames before transformation.

Rules applied here:
- Remove refunded transactions (refund_flag = 1)
- Drop transactions with null revenue or product_id
- Remove negative revenue rows (data artifacts)
- Normalize channel names to title case
- Assign campaign_id=0 to "organic / no campaign" traffic
"""

import pandas as pd


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["refund_flag"] == 0].copy()
    df = df.dropna(subset=["gross_revenue", "product_id"])
    df = df[df["gross_revenue"] > 0]
    df = df.reset_index(drop=True)
    return df


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    # device_type has ~40k nulls — label them "unknown" rather than drop
    df = df.copy()
    df["device_type"] = df["device_type"].fillna("Unknown")
    # campaign_id NaN means organic/direct — already 0 in raw but enforce
    df["campaign_id"] = df["campaign_id"].fillna(0).astype(int)
    return df


def clean_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["channel"] = df["channel"].str.strip().str.title()
    return df


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["country"] = df["country"].str.strip().str.upper()
    df["gender"] = df["gender"].str.strip().str.title()
    df["loyalty_tier"] = df["loyalty_tier"].str.strip().str.title()
    return df


def clean_all(raw: dict) -> dict:
    """Apply all cleaning steps and return a cleaned dict of DataFrames."""
    return {
        "transactions": clean_transactions(raw["transactions"]),
        "customers": clean_customers(raw["customers"]),
        "events": clean_events(raw["events"]),
        "campaigns": clean_campaigns(raw["campaigns"]),
        "products": raw["products"].copy(),
    }
