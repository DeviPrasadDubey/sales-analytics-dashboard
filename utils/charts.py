from __future__ import annotations

import pandas as pd
import plotly.express as px


def revenue_trend(df: pd.DataFrame):
    """Return a line chart of revenue over time."""
    trend_df = (
        df.groupby(df["date"].dt.to_period("M").dt.to_timestamp(), as_index=False)["revenue"]
        .sum()
        .sort_values("date")
    )
    return px.line(
        trend_df,
        x="date",
        y="revenue",
        title="Revenue Trend",
        labels={"date": "Month", "revenue": "Revenue"},
    )


def top_products(df: pd.DataFrame, limit: int = 10):
    """Return a bar chart of top products by revenue."""
    product_df = (
        df.groupby("product", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .head(limit)
    )
    return px.bar(
        product_df,
        x="product",
        y="revenue",
        title=f"Top {limit} Products by Revenue",
        labels={"product": "Product", "revenue": "Revenue"},
    )


def region_breakdown(df: pd.DataFrame):
    """Return a pie chart of revenue split by region."""
    region_df = df.groupby("region", as_index=False)["revenue"].sum()
    return px.pie(
        region_df,
        names="region",
        values="revenue",
        title="Revenue by Region",
    )


def region_profit_comparison(df: pd.DataFrame):
    """Return a bar chart comparing regions by total profit."""
    profit_df = (
        df.groupby("region", as_index=False)["profit"]
        .sum()
        .sort_values("profit", ascending=False)
    )
    return px.bar(
        profit_df,
        x="region",
        y="profit",
        title="Profit by Region",
        labels={"region": "Region", "profit": "Profit"},
    )
