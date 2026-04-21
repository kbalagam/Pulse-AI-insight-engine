"""
Transforms cleaned DataFrames into the 4 analytical target tables.

Target tables:
    fact_daily_metrics      - grain: date
    fact_product_sales      - grain: date x product_id
    fact_marketing_channel  - grain: date x channel
    dim_events              - grain: date x event_type

Spend simulation rationale:
    The campaigns table has no daily spend column. Spend is simulated by
    distributing a channel-realistic daily budget across each campaign's
    active days. Channel base rates (USD/day) are grounded in typical
    SMB benchmarks and scaled proportionally — not intended to be exact,
    but sufficient for trend and ROAS analysis in a demo context.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


CHANNEL_DAILY_BUDGET = {
    "Paid Search": 450,
    "Social": 300,
    "Display": 180,
    "Email": 80,
    "Affiliate": 120,
}


# ---------------------------------------------------------------------------
# Spend simulation
# ---------------------------------------------------------------------------

def _build_daily_spend(campaigns: pd.DataFrame) -> pd.DataFrame:
    """
    Expands each campaign into one row per active day with a daily spend value.
    Returns a DataFrame with columns: date, channel, campaign_id, daily_spend.
    """
    records = []
    for _, row in campaigns.iterrows():
        base_rate = CHANNEL_DAILY_BUDGET.get(row["channel"], 100)
        # Scale by expected_uplift so higher-ROI campaigns have larger budgets
        daily_spend = base_rate * (1 + row["expected_uplift"])
        current = row["start_date"].date()
        end = row["end_date"].date()
        while current <= end:
            records.append({
                "date": current,
                "channel": row["channel"],
                "campaign_id": row["campaign_id"],
                "daily_spend": round(daily_spend, 2),
            })
            current += timedelta(days=1)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# fact_daily_metrics
# ---------------------------------------------------------------------------

def build_fact_daily_metrics(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    events: pd.DataFrame,
    daily_spend: pd.DataFrame,
) -> pd.DataFrame:

    txn = transactions.copy()
    txn["date"] = pd.to_datetime(txn["date"])

    # Revenue and order aggregates
    daily_txn = txn.groupby("date").agg(
        revenue=("gross_revenue", "sum"),
        orders=("transaction_id", "count"),
        customers=("customer_id", "nunique"),
    ).reset_index()

    # New customers: signup_date matches transaction date
    cust = customers[["customer_id", "signup_date_only"]].copy()
    cust["signup_date_only"] = pd.to_datetime(cust["signup_date_only"])
    txn_with_signup = txn.merge(cust, on="customer_id", how="left")
    txn_with_signup["is_new"] = (
        txn_with_signup["date"] == txn_with_signup["signup_date_only"]
    )
    new_cust = txn_with_signup[txn_with_signup["is_new"]].groupby("date").agg(
        new_customers=("customer_id", "nunique")
    ).reset_index()

    daily_txn = daily_txn.merge(new_cust, on="date", how="left")
    daily_txn["new_customers"] = daily_txn["new_customers"].fillna(0).astype(int)
    daily_txn["returning_customers"] = (
        daily_txn["customers"] - daily_txn["new_customers"]
    )

    # Click and conversion counts from events
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    clicks = ev[ev["event_type"] == "click"].groupby("date").size().reset_index(name="clicks")
    conversions = ev[ev["event_type"] == "purchase"].groupby("date").size().reset_index(name="conversions")

    daily_txn = daily_txn.merge(clicks, on="date", how="left")
    daily_txn = daily_txn.merge(conversions, on="date", how="left")
    daily_txn["clicks"] = daily_txn["clicks"].fillna(0).astype(int)
    daily_txn["conversions"] = daily_txn["conversions"].fillna(0).astype(int)

    # Total daily spend
    spend_agg = daily_spend.groupby("date").agg(
        spend=("daily_spend", "sum")
    ).reset_index()
    spend_agg["date"] = pd.to_datetime(spend_agg["date"])
    daily_txn = daily_txn.merge(spend_agg, on="date", how="left")
    daily_txn["spend"] = daily_txn["spend"].fillna(0).round(2)

    # Derived KPIs — guard against division by zero
    daily_txn["cac"] = np.where(
        daily_txn["new_customers"] > 0,
        (daily_txn["spend"] / daily_txn["new_customers"]).round(2),
        np.nan,
    )
    daily_txn["conversion_rate"] = np.where(
        daily_txn["clicks"] > 0,
        (daily_txn["conversions"] / daily_txn["clicks"]).round(4),
        np.nan,
    )
    daily_txn["aov"] = (daily_txn["revenue"] / daily_txn["orders"]).round(2)
    daily_txn["roas"] = np.where(
        daily_txn["spend"] > 0,
        (daily_txn["revenue"] / daily_txn["spend"]).round(2),
        np.nan,
    )

    daily_txn = daily_txn.sort_values("date").reset_index(drop=True)
    return daily_txn


# ---------------------------------------------------------------------------
# fact_product_sales
# ---------------------------------------------------------------------------

def build_fact_product_sales(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:

    txn = transactions.copy()
    txn["date"] = pd.to_datetime(txn["date"])

    prod_sales = txn.groupby(["date", "product_id"]).agg(
        units_sold=("quantity", "sum"),
        revenue=("gross_revenue", "sum"),
        avg_discount=("discount_applied", "mean"),
    ).reset_index()

    prod_meta = products[["product_id", "product_label", "category", "brand", "base_price"]].copy()
    prod_sales = prod_sales.merge(prod_meta, on="product_id", how="left")

    prod_sales["profit"] = (
        prod_sales["revenue"] * (1 - prod_sales["avg_discount"])
    ).round(2)

    prod_sales = prod_sales[[
        "date", "product_id", "product_label", "category", "brand",
        "units_sold", "revenue", "profit",
    ]].sort_values(["date", "product_id"]).reset_index(drop=True)

    return prod_sales


# ---------------------------------------------------------------------------
# fact_marketing_channel
# ---------------------------------------------------------------------------

def build_fact_marketing_channel(
    transactions: pd.DataFrame,
    events: pd.DataFrame,
    campaigns: pd.DataFrame,
    daily_spend: pd.DataFrame,
) -> pd.DataFrame:

    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"])

    # Map events to channel via campaign_id
    camp_channel = campaigns[["campaign_id", "channel"]].copy()
    ev = ev.merge(camp_channel, on="campaign_id", how="left")
    ev["channel"] = ev["channel"].fillna("Organic / Direct")

    clicks = (
        ev[ev["event_type"] == "click"]
        .groupby(["date", "channel"]).size()
        .reset_index(name="clicks")
    )
    conversions = (
        ev[ev["event_type"] == "purchase"]
        .groupby(["date", "channel"]).size()
        .reset_index(name="conversions")
    )

    channel_events = clicks.merge(conversions, on=["date", "channel"], how="outer").fillna(0)
    channel_events["clicks"] = channel_events["clicks"].astype(int)
    channel_events["conversions"] = channel_events["conversions"].astype(int)

    # Revenue attributed via transaction campaign_id → channel
    txn = transactions.copy()
    txn["date"] = pd.to_datetime(txn["date"])
    txn = txn.merge(camp_channel, on="campaign_id", how="left")
    txn["channel"] = txn["channel"].fillna("Organic / Direct")
    revenue_attr = txn.groupby(["date", "channel"]).agg(
        revenue_attributed=("gross_revenue", "sum")
    ).reset_index()

    # Spend from simulation
    spend_ch = daily_spend.groupby(["date", "channel"]).agg(
        spend=("daily_spend", "sum")
    ).reset_index()
    spend_ch["date"] = pd.to_datetime(spend_ch["date"])

    channel_df = channel_events.merge(spend_ch, on=["date", "channel"], how="left")
    channel_df = channel_df.merge(revenue_attr, on=["date", "channel"], how="left")
    channel_df["spend"] = channel_df["spend"].fillna(0).round(2)
    channel_df["revenue_attributed"] = channel_df["revenue_attributed"].fillna(0).round(2)

    channel_df = channel_df.sort_values(["date", "channel"]).reset_index(drop=True)
    return channel_df


# ---------------------------------------------------------------------------
# dim_events
# ---------------------------------------------------------------------------

def build_dim_events(
    events: pd.DataFrame,
    campaigns: pd.DataFrame,
) -> pd.DataFrame:

    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"])

    camp_meta = campaigns[["campaign_id", "channel", "expected_uplift"]].copy()
    ev = ev.merge(camp_meta, on="campaign_id", how="left")

    EVENT_LABELS = {
        "view": "Product View",
        "click": "Link / Ad Click",
        "add_to_cart": "Add to Cart",
        "bounce": "Session Bounce",
        "purchase": "Completed Purchase",
    }

    dim = ev.groupby(["date", "event_type"]).agg(
        event_count=("event_id", "count"),
        avg_session_duration_sec=("session_duration_sec", "mean"),
        expected_impact=("expected_uplift", "mean"),
    ).reset_index()

    dim["event_name"] = dim["event_type"].map(EVENT_LABELS)
    dim["avg_session_duration_sec"] = dim["avg_session_duration_sec"].round(1)
    dim["expected_impact"] = dim["expected_impact"].round(4)

    dim = dim[[
        "date", "event_type", "event_name", "event_count",
        "avg_session_duration_sec", "expected_impact",
    ]].sort_values(["date", "event_type"]).reset_index(drop=True)

    return dim


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_all(cleaned: dict) -> dict:
    """Build all 4 target tables from the cleaned source dict."""
    daily_spend = _build_daily_spend(cleaned["campaigns"])

    return {
        "fact_daily_metrics": build_fact_daily_metrics(
            cleaned["transactions"],
            cleaned["customers"],
            cleaned["events"],
            daily_spend,
        ),
        "fact_product_sales": build_fact_product_sales(
            cleaned["transactions"],
            cleaned["products"],
        ),
        "fact_marketing_channel": build_fact_marketing_channel(
            cleaned["transactions"],
            cleaned["events"],
            cleaned["campaigns"],
            daily_spend,
        ),
        "dim_events": build_dim_events(
            cleaned["events"],
            cleaned["campaigns"],
        ),
    }
