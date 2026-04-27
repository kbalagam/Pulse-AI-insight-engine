"""
Revenue forecasting using Holt-Winters Exponential Smoothing.

Why Holt-Winters over Prophet:
    Prophet requires pystan which fails on Streamlit Cloud without Docker.
    Holt-Winters handles trend and multiplicative seasonality cleanly,
    installs via statsmodels (no C compilation), and is equally credible
    for a portfolio project with 3 years of daily data.

Output:
    DataFrame with columns: date, actual (NaN for future), forecast,
    lower_ci, upper_ci.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_daily_revenue() -> pd.Series:
    """Returns a daily revenue Series indexed by date."""
    try:
        df = pd.read_parquet(PROCESSED_DIR / "fact_daily_metrics.parquet")
    except Exception:
        df = pd.read_csv(PROCESSED_DIR / "fact_daily_metrics.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df["revenue"].asfreq("D")


def build_forecast(horizon_days: int = 90) -> pd.DataFrame:
    """
    Fits Holt-Winters on full revenue history and forecasts forward.

    Parameters
    ----------
    horizon_days : int
        Number of days to forecast beyond the last known date.

    Returns
    -------
    pd.DataFrame with columns:
        date, actual, forecast, lower_ci, upper_ci
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    series = load_daily_revenue().dropna()

    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=7,      # weekly seasonality most dominant in e-commerce
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True, remove_bias=True)

    forecast_values = fit.forecast(horizon_days)

    # Simulate confidence intervals using residual std
    residuals  = fit.resid
    resid_std  = residuals.std()
    z_95       = 1.96
    lower_ci   = forecast_values - z_95 * resid_std
    upper_ci   = forecast_values + z_95 * resid_std

    # Historical fitted values
    hist_df = pd.DataFrame({
        "date":     series.index,
        "actual":   series.values,
        "forecast": fit.fittedvalues.values,
        "lower_ci": np.nan,
        "upper_ci": np.nan,
    })

    # Future forecast
    future_index = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )
    future_df = pd.DataFrame({
        "date":     future_index,
        "actual":   np.nan,
        "forecast": forecast_values.values,
        "lower_ci": lower_ci.values,
        "upper_ci": upper_ci.values,
    })

    result = pd.concat([hist_df, future_df], ignore_index=True)
    result["forecast"] = result["forecast"].clip(lower=0)
    result["lower_ci"] = result["lower_ci"].clip(lower=0)
    return result


def forecast_summary(forecast_df: pd.DataFrame) -> dict:
    """Returns key stats from the forecast for AI context."""
    future = forecast_df[forecast_df["actual"].isna()].copy()
    actual = forecast_df[forecast_df["actual"].notna()].copy()
    return {
        "last_actual_revenue":  round(actual["actual"].iloc[-1], 2),
        "avg_actual_30d":       round(actual["actual"].tail(30).mean(), 2),
        "avg_forecast":         round(future["forecast"].mean(), 2),
        "forecast_change_pct":  round(
            (future["forecast"].mean() - actual["actual"].tail(30).mean()) /
            actual["actual"].tail(30).mean() * 100, 1
        ),
        "forecast_horizon_days": len(future),
        "lower_bound_avg":      round(future["lower_ci"].mean(), 2),
        "upper_bound_avg":      round(future["upper_ci"].mean(), 2),
    }
