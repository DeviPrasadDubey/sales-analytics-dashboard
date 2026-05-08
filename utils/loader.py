from __future__ import annotations

import pandas as pd


def load_sales_data(csv_path: str) -> pd.DataFrame:
    """Load sales CSV, parse dates, and compute profit."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    numeric_cols = ["units_sold", "revenue", "cost"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()
    df["profit"] = df["revenue"] - df["cost"]
    return df
