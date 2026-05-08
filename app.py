from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

from utils import charts
from utils.ai_analyst import get_ai_insight
from utils.loader import load_sales_data


st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "sample_sales.csv"

st.title("Sales Analytics Dashboard")

df = load_sales_data(str(DATA_PATH))
df["date"] = pd.to_datetime(df["date"])
min_date = df["date"].min()
max_date = df["date"].max()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
if len(date_range) == 2:
    df = df[(df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))]

page = st.sidebar.radio("Navigation", ["Overview", "Products", "Regions", "Forecast", "AI Analyst"])

if df.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

if page == "Overview":
    st.subheader("Overview")
    total_revenue = float(df["revenue"].sum())
    total_profit = float(df["profit"].sum())
    total_orders = int(len(df))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Profit", f"${total_profit:,.2f}")
    col3.metric("Total Orders", f"{total_orders:,}")
    fig = charts.revenue_trend(df)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Products":
    st.subheader("Products")
    fig = charts.top_products(df)
    st.plotly_chart(fig, use_container_width=True)
    product_summary = df.groupby("product").agg(
        total_revenue=("revenue", "sum"),
        total_units=("units_sold", "sum"),
        total_profit=("profit", "sum"),
    ).reset_index()
    product_summary["profit_margin_%"] = (product_summary["total_profit"] / product_summary["total_revenue"] * 100).round(2)
    st.dataframe(product_summary.sort_values("total_revenue", ascending=False))

elif page == "Regions":
    st.subheader("Regions")
    fig = charts.region_breakdown(df)
    st.plotly_chart(fig, use_container_width=True)
    region_profit = df.groupby("region")["profit"].sum().reset_index()
    fig2 = px.bar(region_profit, x="region", y="profit", title="Profit by Region")
    st.plotly_chart(fig2, use_container_width=True)
    best = region_profit.loc[region_profit["profit"].idxmax(), "region"]
    worst = region_profit.loc[region_profit["profit"].idxmin(), "region"]
    avg_order = df["revenue"].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Region", best)
    col2.metric("Worst Region", worst)
    col3.metric("Avg Order Value", f"₹{avg_order:,.0f}")

elif page == "Forecast":
    st.subheader("Forecast")
    daily_revenue = df.groupby("date")["revenue"].sum().reset_index()
    daily_revenue = daily_revenue.sort_values("date")
    daily_revenue["day_index"] = (daily_revenue["date"] - daily_revenue["date"].min()).dt.days

    X = daily_revenue[["day_index"]]
    y = daily_revenue["revenue"]
    model = LinearRegression()
    model.fit(X, y)

    last_day = daily_revenue["day_index"].max()
    future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
    future_dates = pd.date_range(daily_revenue["date"].max() + pd.Timedelta(days=1), periods=30)
    future_revenue = model.predict(future_days)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_revenue["date"],
            y=daily_revenue["revenue"],
            name="Actual Revenue",
            line=dict(color="#2eb89a"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=future_revenue,
            name="Forecast",
            line=dict(color="orange", dash="dash"),
        )
    )
    fig.update_layout(title="Revenue Forecast — Next 30 Days", xaxis_title="Date", yaxis_title="Revenue")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecasted Revenue (30d)", f"₹{future_revenue.sum():,.0f}")
    col2.metric("Daily Growth Rate", f"₹{model.coef_[0]:,.0f}/day")
    col3.metric("Peak Forecast Day", f"₹{future_revenue.max():,.0f}")

elif page == "AI Analyst":
    st.subheader("AI Analyst")
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    if uploaded_file is not None:
        ai_df = pd.read_csv(uploaded_file)
        st.write("Columns:", ai_df.columns.tolist())
        st.write("Shape:", ai_df.shape)
        question = st.text_input("Ask anything about your data")
        if question:
            response = get_ai_insight(ai_df, question)
            st.info(response)
        st.dataframe(ai_df.head(10))
