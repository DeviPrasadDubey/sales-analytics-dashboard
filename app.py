from pathlib import Path

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xlsxwriter
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.ai_analyst import get_ai_insight
from utils.loader import load_sales_data


def detect_columns(df):
    """Auto-detect column types from any dataframe"""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "date": date_cols,
        "first_numeric": numeric_cols[0] if numeric_cols else None,
        "second_numeric": numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] if numeric_cols else None,
        "first_cat": categorical_cols[0] if categorical_cols else None,
        "second_cat": categorical_cols[1] if len(categorical_cols) > 1 else None,
        "first_date": date_cols[0] if date_cols else None,
    }


def export_to_excel(df):
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True})
    worksheet = workbook.add_worksheet("Data")
    
    # Header format
    header_format = workbook.add_format({
        "bold": True,
        "bg_color": "#2eb89a",
        "font_color": "white",
        "border": 1
    })
    
    # Write headers
    for col_num, col_name in enumerate(df.columns):
        worksheet.write(0, col_num, col_name, header_format)
    
    # Write data
    for row_num, row in enumerate(df.values):
        for col_num, value in enumerate(row):
            worksheet.write(row_num + 1, col_num, str(value))
    
    workbook.close()
    output.seek(0)
    return output.getvalue()


def export_summary_excel(df):
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True})
    
    cols = detect_columns(df)
    
    # Sheet 1 - Raw Data
    ws1 = workbook.add_worksheet("Raw Data")
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#2eb89a", "font_color": "white"})
    for i, col in enumerate(df.columns):
        ws1.write(0, i, col, header_fmt)
    for row_i, row in enumerate(df.values):
        for col_i, val in enumerate(row):
            ws1.write(row_i + 1, col_i, str(val))
    
    # Sheet 2 - Summary Stats
    ws2 = workbook.add_worksheet("Summary")
    summary = df.describe().reset_index()
    for i, col in enumerate(summary.columns):
        ws2.write(0, i, col, header_fmt)
    for row_i, row in enumerate(summary.values):
        for col_i, val in enumerate(row):
            ws2.write(row_i + 1, col_i, str(val))
    
    # Sheet 3 - KPIs
    if cols["first_cat"] and cols["first_numeric"]:
        ws3 = workbook.add_worksheet("KPI Summary")
        kpi_data = df.groupby(cols["first_cat"])[cols["numeric"]].sum().reset_index()
        for i, col in enumerate(kpi_data.columns):
            ws3.write(0, i, col, header_fmt)
        for row_i, row in enumerate(kpi_data.values):
            for col_i, val in enumerate(row):
                ws3.write(row_i + 1, col_i, str(val))
    
    workbook.close()
    output.seek(0)
    return output.getvalue()


st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "sample_sales.csv"

st.title("Sales Analytics Dashboard")

st.sidebar.markdown("### 📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload Your Data", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    
    if file_ext == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_ext in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded_file)
    elif file_ext == "json":
        df = pd.read_json(uploaded_file)
    
    # Auto detect and parse date columns
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    st.sidebar.success(f"✅ {uploaded_file.name} — {df.shape[0]} rows, {df.shape[1]} cols")
else:
    df = load_sales_data("data/sample_sales.csv")
    st.sidebar.info("📊 Using sample data")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Products",
        "Regions",
        "Forecast",
        "Correlation & Insights",
        "ML Models",
        "AI Analyst",
        "Data Cleaning",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Data")

export_type = st.sidebar.radio("Export format", ["Raw Data (Excel)", "Full Report (Excel)"])

if st.sidebar.button("⬇️ Download"):
    if export_type == "Raw Data (Excel)":
        excel_data = export_to_excel(df)
        st.sidebar.download_button(
            label="📊 Click to Download",
            data=excel_data,
            file_name="data_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        excel_data = export_summary_excel(df)
        st.sidebar.download_button(
            label="📊 Click to Download", 
            data=excel_data,
            file_name="full_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if df.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

if page == "Overview":
    st.subheader("Overview")
    cols = detect_columns(df)

    # KPI Cards - show top 4 numeric columns
    st.markdown("### 📊 Key Metrics")
    numeric_cols = cols["numeric"][:4]
    kpi_cols = st.columns(len(numeric_cols)) if numeric_cols else st.columns(1)
    for i, col in enumerate(numeric_cols):
        kpi_cols[i].metric(col.replace("_", " ").title(), f"{df[col].sum():,.2f}")

    # Date filter in sidebar
    if cols["first_date"]:
        date_col = cols["first_date"]
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        date_range = st.sidebar.date_input("📅 Date Range", [min_date, max_date])
        if len(date_range) == 2:
            df = df[(df[date_col] >= pd.Timestamp(date_range[0])) & 
                    (df[date_col] <= pd.Timestamp(date_range[1]))]

    # Trend chart
    if cols["first_date"] and cols["first_numeric"]:
        trend = df.groupby(cols["first_date"])[cols["first_numeric"]].sum().reset_index()
        fig = px.line(trend, x=cols["first_date"], y=cols["first_numeric"], 
                      title=f"{cols['first_numeric'].replace('_',' ').title()} Over Time",
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)
    elif cols["first_cat"] and cols["first_numeric"]:
        summary = df.groupby(cols["first_cat"])[cols["first_numeric"]].sum().reset_index()
        fig = px.bar(summary, x=cols["first_cat"], y=cols["first_numeric"],
                     title=f"{cols['first_numeric']} by {cols['first_cat']}", color=cols["first_cat"])
        st.plotly_chart(fig, use_container_width=True)

elif page == "Products":
    st.subheader("Products")
    cols = detect_columns(df)
    if cols["first_cat"] and cols["first_numeric"]:
        cat_col = cols["first_cat"]
        num_col = cols["first_numeric"]
        
        # Let user pick columns
        selected_cat = st.selectbox("Group by", cols["categorical"])
        selected_num = st.selectbox("Measure", cols["numeric"])
        
        summary = df.groupby(selected_cat)[selected_num].sum().reset_index().sort_values(selected_num, ascending=False)
        
        fig = px.bar(summary.head(10), x=selected_cat, y=selected_num, 
                     title=f"Top 10 {selected_cat} by {selected_num}", color=selected_cat)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(summary)
    else:
        st.warning("Upload data with at least one categorical and one numeric column")

elif page == "Regions":
    st.subheader("Regions")
    cols = detect_columns(df)
    if cols["first_cat"] and cols["first_numeric"]:
        selected_cat = st.selectbox("Segment by", cols["categorical"])
        selected_num = st.selectbox("Metric", cols["numeric"])
        
        summary = df.groupby(selected_cat)[selected_num].sum().reset_index()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(summary, names=selected_cat, values=selected_num, 
                         title=f"{selected_num} Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(summary, x=selected_cat, y=selected_num,
                          title=f"{selected_num} by {selected_cat}", color=selected_cat)
            st.plotly_chart(fig2, use_container_width=True)
        
        best = summary.loc[summary[selected_num].idxmax(), selected_cat]
        worst = summary.loc[summary[selected_num].idxmin(), selected_cat]
        avg = df[selected_num].mean()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Best", best)
        c2.metric("Worst", worst)
        c3.metric("Average", f"{avg:,.2f}")
    else:
        st.warning("Upload data with categorical and numeric columns")

elif page == "Forecast":
    st.subheader("Forecast")
    cols = detect_columns(df)
    if cols["first_date"] and cols["first_numeric"]:
        date_col = cols["first_date"]
        selected_num = st.selectbox("Forecast metric", cols["numeric"])
        forecast_days = st.slider("Forecast days", 7, 90, 30)
        
        daily = df.groupby(date_col)[selected_num].sum().reset_index().sort_values(date_col)
        daily["day_index"] = (daily[date_col] - daily[date_col].min()).dt.days
        
        X = daily[["day_index"]]
        y = daily[selected_num]
        model = LinearRegression()
        model.fit(X, y)
        
        last_day = daily["day_index"].max()
        future_days = np.arange(last_day + 1, last_day + forecast_days + 1).reshape(-1, 1)
        future_dates = pd.date_range(daily[date_col].max() + pd.Timedelta(days=1), periods=forecast_days)
        future_values = model.predict(future_days)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily[date_col], y=daily[selected_num], name="Actual", line=dict(color="#2eb89a")))
        fig.add_trace(go.Scatter(x=future_dates, y=future_values, name="Forecast", line=dict(color="orange", dash="dash")))
        fig.update_layout(title=f"{selected_num} Forecast — Next {forecast_days} Days")
        st.plotly_chart(fig, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Forecasted {selected_num} ({forecast_days}d)", f"{future_values.sum():,.0f}")
        c2.metric("Daily Growth", f"{model.coef_[0]:,.2f}/day")
        c3.metric("Peak Day", f"{future_values.max():,.0f}")
    else:
        st.warning("Forecast needs a date column and numeric column in your data")

elif page == "Correlation & Insights":
    st.title("🔗 Correlation & Insights")
    st.markdown("Discover hidden relationships between variables in your data")
    
    cols = detect_columns(df)
    
    if len(cols["numeric"]) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis")
    else:
        # ── SECTION 1: Correlation Heatmap ──────────────────────
        st.markdown("## 🌡️ Correlation Heatmap")
        st.info("💡 Correlation shows how strongly two columns are related. Value close to 1 means when one goes up, other goes up too. Value close to -1 means opposite relationship. Value near 0 means no relationship.")
        
        corr_matrix = df[cols["numeric"]].corr().round(2)
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdYlGn",
            title="Correlation Matrix — All Numeric Columns",
            aspect="auto",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Strongest correlations
        st.markdown("### 🔍 Strongest Relationships")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                val = corr_matrix.iloc[i, j]
                corr_pairs.append({
                    "Column 1": col1,
                    "Column 2": col2,
                    "Correlation": val,
                    "Strength": "Strong" if abs(val) > 0.7 else "Moderate" if abs(val) > 0.4 else "Weak",
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values("Correlation", key=abs, ascending=False)
        st.dataframe(corr_df)

        # ── SECTION 2: Scatter Plot ──────────────────────────────
        st.markdown("## 📍 Scatter Plot Explorer")
        st.info("💡 Scatter plot shows relationship between two numeric columns as dots. If dots form a line going up — positive correlation. Going down — negative correlation. Random dots — no correlation.")
        
        col1_select = st.selectbox("X axis", cols["numeric"], key="scatter_x")
        col2_select = st.selectbox("Y axis", cols["numeric"], index=1 if len(cols["numeric"]) > 1 else 0, key="scatter_y")
        color_by = st.selectbox("Color by (optional)", ["None"] + cols["categorical"], key="scatter_color")
        
        if color_by == "None":
            fig2 = px.scatter(
                df,
                x=col1_select,
                y=col2_select,
                title=f"{col1_select} vs {col2_select}",
                trendline="ols",
            )
        else:
            fig2 = px.scatter(
                df,
                x=col1_select,
                y=col2_select,
                color=color_by,
                title=f"{col1_select} vs {col2_select} by {color_by}",
                trendline="ols",
            )
        st.plotly_chart(fig2, use_container_width=True)

        # ── SECTION 3: Distribution Analysis ────────────────────
        st.markdown("## 📊 Distribution Analysis")
        st.info("💡 Distribution shows how your data is spread. A bell curve means data is normally distributed — most values near average. Skewed means most values are on one side.")
        
        dist_col = st.selectbox("Select column", cols["numeric"], key="dist_col")
        
        col_a, col_b = st.columns(2)
        with col_a:
            fig3 = px.histogram(
                df,
                x=dist_col,
                title=f"Distribution of {dist_col}",
                nbins=30,
                color_discrete_sequence=["#2eb89a"],
            )
            st.plotly_chart(fig3, use_container_width=True)
        with col_b:
            fig4 = px.box(
                df,
                y=dist_col,
                title=f"Boxplot of {dist_col}",
                color_discrete_sequence=["#2eb89a"],
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        # Stats
        col_stats = df[dist_col]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mean", f"{col_stats.mean():,.2f}")
        s2.metric("Median", f"{col_stats.median():,.2f}")
        s3.metric("Std Dev", f"{col_stats.std():,.2f}")
        s4.metric("Skewness", f"{col_stats.skew():,.2f}")

        # ── SECTION 4: Auto Insights ─────────────────────────────
        st.markdown("## 💡 Auto Insights")
        st.info("💡 These are automatically generated observations about your data — no manual analysis needed.")
        
        insights = []
        
        # Strongest correlation insight
        if len(corr_df) > 0:
            top = corr_df.iloc[0]
            direction = "positively" if top["Correlation"] > 0 else "negatively"
            insights.append(
                f"🔗 **{top['Column 1']}** and **{top['Column 2']}** are most {direction} correlated ({top['Correlation']}) — when one increases, the other {'increases' if top['Correlation'] > 0 else 'decreases'} too."
            )
        
        # Missing values insight
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100).round(2)
        if missing_pct > 0:
            insights.append(f"⚠️ Your dataset has **{missing_pct}% missing values** overall. Consider cleaning before analysis.")
        else:
            insights.append("✅ Your dataset has **no missing values** — great data quality!")
        
        # Size insight
        insights.append(
            f"📦 Dataset has **{df.shape[0]:,} rows** and **{df.shape[1]} columns** — {'sufficient' if df.shape[0] > 100 else 'small sample, insights may not be reliable'} for analysis."
        )
        
        # Numeric columns insight
        for col in cols["numeric"][:2]:
            skew = df[col].skew()
            if abs(skew) > 1:
                insights.append(
                    f"📈 **{col}** is highly skewed ({skew:.2f}) — there are extreme values pulling the average. Median is more reliable than mean here."
                )
        
        for insight in insights:
            st.markdown(insight)

elif page == "ML Models":
    st.title("🤖 ML Models")
    st.markdown("Train and evaluate machine learning models on your data — no coding needed")

    cols = detect_columns(df)

    if len(cols["numeric"]) < 2:
        st.warning("Need at least 2 numeric columns to run ML models")
    else:
        # ── MODEL SELECTION ──────────────────────────────────────
        st.markdown("## 🎯 Select Problem Type")
        st.info("💡 Not sure which to pick? Regression = predict a number. Classification = predict a category. Clustering = find hidden groups.")

        problem_type = st.radio(
            "What do you want to do?",
            [
                "📈 Regression — Predict a numeric value",
                "🏷️ Classification — Predict a category",
                "🔵 Clustering — Find groups in data",
            ],
        )

        st.markdown("---")

        # ══ REGRESSION ══════════════════════════════════════════
        if "Regression" in problem_type:
            st.markdown("## 📈 Regression")
            st.info("💡 Regression predicts a number — like house price, revenue, or temperature. You pick which column to predict (target) and which columns to use as inputs (features).")

            target = st.selectbox("Target column (what to predict)", cols["numeric"])
            features = st.multiselect("Feature columns (inputs)", [c for c in cols["numeric"] if c != target])

            model_choice = st.selectbox("Model", ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"])
            test_size = st.slider("Test data %", 10, 40, 20)

            if st.button("Train Regression Model") and features:
                X = df[features].dropna()
                y = df.loc[X.index, target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Decision Tree Regressor":
                    max_depth = st.session_state.get("dt_depth", 5)
                    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                st.markdown("### 📊 Model Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("R² Score", f"{r2:.4f}")
                c2.metric("RMSE", f"{rmse:,.2f}")
                c3.metric("Training Samples", len(X_train))

                st.info(
                    f"💡 R² of {r2:.2f} means the model explains {r2 * 100:.1f}% of the variation in {target}. Higher is better. RMSE of {rmse:,.2f} is the average prediction error."
                )

                # Actual vs Predicted chart
                pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
                fig = px.scatter(
                    pred_df,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs Predicted",
                    trendline="ols",
                    color_discrete_sequence=["#2eb89a"],
                )
                fig.add_shape(
                    type="line",
                    x0=float(y_test.min()),
                    y0=float(y_test.min()),
                    x1=float(y_test.max()),
                    y1=float(y_test.max()),
                    line=dict(color="red", dash="dash"),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Feature importance
                if hasattr(model, "feature_importances_"):
                    st.markdown("### 🔍 Feature Importance")
                    st.info("💡 Feature importance shows which input columns matter most for prediction.")
                    imp_df = (
                        pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
                        .sort_values("Importance", ascending=False)
                    )
                    fig2 = px.bar(imp_df, x="Feature", y="Importance", title="Feature Importance", color="Feature")
                    st.plotly_chart(fig2, use_container_width=True)
                elif model_choice == "Linear Regression":
                    st.markdown("### 🔍 Feature Coefficients")
                    st.info("💡 Coefficients show how much target changes when feature increases by 1.")
                    coef_df = (
                        pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
                        .sort_values("Coefficient", ascending=False)
                    )
                    fig2 = px.bar(coef_df, x="Feature", y="Coefficient", title="Feature Coefficients", color="Feature")
                    st.plotly_chart(fig2, use_container_width=True)

                # Residuals
                st.markdown("### 📉 Residual Plot")
                st.info("💡 Residuals are prediction errors. Good model = residuals randomly scattered around 0.")
                residuals = y_test.values - y_pred
                fig3 = px.scatter(
                    x=y_pred,
                    y=residuals,
                    title="Residuals vs Predicted",
                    labels={"x": "Predicted", "y": "Residual"},
                    color_discrete_sequence=["#2eb89a"],
                )
                fig3.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig3, use_container_width=True)

        # ══ CLASSIFICATION ═══════════════════════════════════════
        elif "Classification" in problem_type:
            st.markdown("## 🏷️ Classification")
            st.info("💡 Classification predicts a category — like Yes/No, High/Medium/Low, or Churn/No Churn. Pick a categorical column to predict and numeric columns as inputs.")

            all_targets = cols["categorical"] + cols["numeric"]
            target = st.selectbox("Target column (what to predict)", all_targets)
            features = st.multiselect("Feature columns (inputs)", cols["numeric"])

            model_choice = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
            test_size = st.slider("Test data %", 10, 40, 20)

            if st.button("Train Classification Model") and features:
                # Encode target
                le = LabelEncoder()
                X = df[features].dropna()
                y_raw = df.loc[X.index, target].astype(str)
                y = le.fit_transform(y_raw)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                if model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000, random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                elif model_choice == "Decision Tree":
                    model = DecisionTreeClassifier(max_depth=5, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)

                st.markdown("### 📊 Model Results")
                c1, c2 = st.columns(2)
                c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
                c2.metric("Test Samples", len(X_test))

                st.info(
                    f"💡 Accuracy of {accuracy * 100:.1f}% means the model correctly predicted {accuracy * 100:.1f} out of every 100 samples."
                )

                # Confusion Matrix
                st.markdown("### 🔲 Confusion Matrix")
                st.info("💡 Confusion matrix shows correct vs incorrect predictions for each category.")
                cm = confusion_matrix(y_test, y_pred)
                class_names = le.classes_
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    x=class_names,
                    y=class_names,
                    title="Confusion Matrix",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Feature Importance
                if hasattr(model, "feature_importances_"):
                    st.markdown("### 🔍 Feature Importance")
                    imp_df = (
                        pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
                        .sort_values("Importance", ascending=False)
                    )
                    fig2 = px.bar(imp_df, x="Feature", y="Importance", title="Feature Importance", color="Feature")
                    st.plotly_chart(fig2, use_container_width=True)

        # ══ CLUSTERING ═══════════════════════════════════════════
        elif "Clustering" in problem_type:
            st.markdown("## 🔵 Clustering — Customer Segmentation")
            st.info(
                "💡 Clustering automatically finds hidden groups in your data — like high value vs low value customers, or different buying behavior segments. No target column needed."
            )

            features = st.multiselect("Select columns for clustering", cols["numeric"])
            n_clusters = st.slider("Number of groups (clusters)", 2, 8, 3)

            if st.button("Run Clustering") and features:
                X = df[features].dropna()

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)

                result_df = X.copy()
                result_df["Cluster"] = clusters.astype(str)

                st.markdown("### 📊 Cluster Results")
                c1, c2 = st.columns(2)
                c1.metric("Clusters Found", n_clusters)
                c2.metric("Samples Clustered", len(result_df))

                # Cluster distribution
                cluster_counts = result_df["Cluster"].value_counts().reset_index()
                cluster_counts.columns = ["Cluster", "Count"]
                fig = px.pie(cluster_counts, names="Cluster", values="Count", title="Cluster Distribution")
                st.plotly_chart(fig, use_container_width=True)

                # Scatter if 2+ features
                if len(features) >= 2:
                    fig2 = px.scatter(
                        result_df,
                        x=features[0],
                        y=features[1],
                        color="Cluster",
                        title=f"Clusters — {features[0]} vs {features[1]}",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Cluster summary
                st.markdown("### 📋 Cluster Profile")
                st.info(
                    "💡 This shows the average values for each cluster — helping you understand what makes each group different."
                )
                cluster_summary = result_df.groupby("Cluster")[features].mean().round(2)
                st.dataframe(cluster_summary)

                # Elbow chart
                st.markdown("### 📉 Elbow Chart")
                st.info(
                    "💡 Elbow chart helps find the optimal number of clusters. Look for the point where the curve bends like an elbow — that's the best cluster count."
                )
                inertias = []
                k_range = range(2, min(10, len(X)))
                for k in k_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)
                fig3 = px.line(
                    x=list(k_range),
                    y=inertias,
                    title="Elbow Chart — Optimal Clusters",
                    labels={"x": "Number of Clusters", "y": "Inertia"},
                    markers=True,
                )
                st.plotly_chart(fig3, use_container_width=True)

                # Export clustered data
                result_df_export = df.loc[X.index].copy()
                result_df_export["Cluster"] = clusters.astype(str)
                excel_data = export_to_excel(result_df_export)
                st.download_button(
                    label="⬇️ Download Clustered Data",
                    data=excel_data,
                    file_name="clustered_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

elif page == "AI Analyst":
    st.subheader("AI Analyst")
    cols = detect_columns(df)
    numeric_cols = cols["numeric"]
    st.write("Columns:", df.columns.tolist())
    st.write("Shape:", df.shape)
    question = st.text_input("Ask anything about your data")
    if question:
        result = get_ai_insight(df, question)
        # Show KPI cards from data
        st.markdown("### 📊 Quick KPIs")
        if numeric_cols:
            kpi_cols = st.columns(min(4, len(numeric_cols)))
            for i, col in enumerate(numeric_cols[:4]):
                kpi_cols[i].metric(col.replace("_", " ").title(), f"{df[col].sum():,.0f}")
        else:
            st.caption("No numeric columns available for KPI cards.")

        st.divider()

        # Show AI text insight
        st.info(result["insight"])

        st.divider()

        # Show chart based on AI response
        chart_spec = result.get("chart", {})
        chart_type = chart_spec.get("type", "none")

        if chart_type != "none":
            chart_x = chart_spec.get("x")
            chart_y = chart_spec.get("y")
            title = chart_spec.get("title", "")
            agg_func = chart_spec.get("agg", "sum")

            try:
                if chart_x and chart_y:
                    if agg_func == "sum":
                        chart_df = df.groupby(chart_x)[chart_y].sum().reset_index()
                    elif agg_func == "mean":
                        chart_df = df.groupby(chart_x)[chart_y].mean().reset_index()
                    elif agg_func == "count":
                        chart_df = df.groupby(chart_x)[chart_y].count().reset_index()
                    else:
                        chart_df = df.groupby(chart_x)[chart_y].sum().reset_index()

                    st.markdown(f"### 📈 {title}")

                    if chart_type == "bar":
                        fig = px.bar(chart_df, x=chart_x, y=chart_y, title=title, color=chart_x)
                    elif chart_type == "line":
                        fig = px.line(chart_df, x=chart_x, y=chart_y, title=title, markers=True)
                    elif chart_type == "pie":
                        fig = px.pie(chart_df, names=chart_x, values=chart_y, title=title)
                    elif chart_type == "scatter":
                        fig = px.scatter(
                            df,
                            x=chart_x,
                            y=chart_y,
                            title=title,
                            color=chart_x if chart_x in df.columns else None,
                        )
                    else:
                        fig = None

                    if fig is not None:
                        fig.update_layout(showlegend=True, height=450)
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Chart generate nahi ho saka: {e}")

    # Always show raw data at bottom
    with st.expander("🔍 Raw Data Preview"):
        st.dataframe(df.head(20))

if page == "Data Cleaning":
    st.title("🧹 Data Cleaning Studio")
    st.markdown("Complete data cleaning and preparation toolkit")

    cols = detect_columns(df)
    
    # Work on a copy
    cleaned_df = df.copy()

    # ── SECTION 1: Overview ──────────────────────────────────────
    st.markdown("## 📋 Dataset Overview")
    st.info(
        "💡 This summary shows how large your dataset is and whether there are obvious issues "
        "(missing cells or duplicated rows). Use it as a quick health check before cleaning."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", cleaned_df.shape[0])
    c2.metric("Total Columns", cleaned_df.shape[1])
    c3.metric("Missing Values", cleaned_df.isnull().sum().sum())
    c4.metric("Duplicate Rows", cleaned_df.duplicated().sum())

    st.dataframe(cleaned_df.head(10))

    # ── SECTION 2: Column Info ───────────────────────────────────
    st.markdown("## 🔬 Column Analysis")
    st.info(
        "💡 Here you can see each column’s type, how many blanks it has, and a sample value. "
        "That helps you spot columns that might need fixing before charts and AI analysis."
    )
    dtype_df = pd.DataFrame({
        "Column": cleaned_df.columns,
        "Data Type": cleaned_df.dtypes.values,
        "Missing Count": cleaned_df.isnull().sum().values,
        "Missing %": (cleaned_df.isnull().sum().values / len(cleaned_df) * 100).round(2),
        "Unique Values": [cleaned_df[col].nunique() for col in cleaned_df.columns],
        "Sample Value": [str(cleaned_df[col].dropna().iloc[0]) if len(cleaned_df[col].dropna()) > 0 else "N/A" for col in cleaned_df.columns]
    })
    st.dataframe(dtype_df)

    # ── SECTION 3: Missing Values ────────────────────────────────
    st.markdown("## ❓ Handle Missing Values")
    st.info("💡 Missing values are empty cells in your data. They can cause errors in analysis. You can either remove rows that have missing values, or fill them with a calculated value like average.")
    missing_cols = cleaned_df.columns[cleaned_df.isnull().any()].tolist()
    
    if missing_cols:
        missing_summary = cleaned_df.isnull().sum().reset_index()
        missing_summary.columns = ["Column", "Missing Count"]
        missing_summary["Missing %"] = (missing_summary["Missing Count"] / len(cleaned_df) * 100).round(2)
        missing_summary = missing_summary[missing_summary["Missing Count"] > 0]
        
        fig = px.bar(missing_summary, x="Column", y="Missing %", 
                     title="Missing Values by Column", color="Column")
        st.plotly_chart(fig, use_container_width=True)

        selected_col = st.selectbox("Select column to fix", missing_cols)
        strategy = st.radio("Fill strategy", [
            "Drop rows with missing values",
            "Fill with Mean (numeric)",
            "Fill with Median (numeric)", 
            "Fill with Mode",
            "Fill with custom value"
        ])
        
        custom_val = ""
        if strategy == "Fill with custom value":
            custom_val = st.text_input("Enter custom value")
        
        if st.button("Apply Missing Value Fix"):
            if strategy == "Drop rows with missing values":
                cleaned_df = cleaned_df.dropna(subset=[selected_col])
                rows_removed = len(df) - len(cleaned_df)
                st.success(f"✅ Done! Removed {rows_removed} rows that had empty values in '{selected_col}'. Your data now has {cleaned_df.shape[0]} rows.")
            elif strategy == "Fill with Mean (numeric)":
                n_empty_mean = cleaned_df[selected_col].isnull().sum()
                mean_val = round(df[selected_col].mean(), 2)
                cleaned_df[selected_col] = cleaned_df[selected_col].fillna(cleaned_df[selected_col].mean())
                st.success(f"✅ Done! Filled {n_empty_mean} empty cells in '{selected_col}' with the average value ({mean_val}). Average means the sum of all values divided by count.")
            elif strategy == "Fill with Median (numeric)":
                median_val = round(df[selected_col].median(), 2)
                cleaned_df[selected_col] = cleaned_df[selected_col].fillna(cleaned_df[selected_col].median())
                st.success(f"✅ Done! Filled empty cells with the middle value ({median_val}). Median is better than mean when your data has very high or very low outliers.")
            elif strategy == "Fill with Mode":
                mode_val = df[selected_col].mode()[0]
                cleaned_df[selected_col] = cleaned_df[selected_col].fillna(cleaned_df[selected_col].mode()[0])
                st.success(f"✅ Done! Filled empty cells with the most common value '{mode_val}'. This works well for text columns like city, category, gender etc.")
            elif strategy == "Fill with custom value" and custom_val:
                cleaned_df[selected_col] = cleaned_df[selected_col].fillna(custom_val)
                st.success(f"✅ Done! Filled all empty cells in '{selected_col}' with your custom value '{custom_val}'.")
            st.dataframe(cleaned_df.head(10))
    else:
        st.success("✅ No missing values found!")

    # ── SECTION 4: Duplicates ────────────────────────────────────
    st.markdown("## 🔁 Duplicate Rows")
    st.info("💡 Duplicate rows are exact copies of other rows in your data. They can make your totals and averages incorrect. For example, if a sale is recorded twice, your total revenue will be wrong.")
    dup_count = cleaned_df.duplicated().sum()
    if dup_count > 0:
        st.warning(f"Found {dup_count} duplicate rows")
        if st.button("Remove Duplicates"):
            cleaned_df = cleaned_df.drop_duplicates()
            st.success(f"✅ Done! Removed {dup_count} duplicate rows. These were exact copies of other rows. Your data is now more accurate with {cleaned_df.shape[0]} rows remaining.")
    else:
        st.success("✅ No duplicates found!")

    # ── SECTION 5: Whitespace & String Cleaning ──────────────────
    st.markdown("## ✂️ String & Whitespace Cleaning")
    st.info("💡 Text columns often have hidden problems — extra spaces before/after words, inconsistent capitalization (like 'Mumbai' vs 'mumbai'), or special characters. These cause grouping errors. For example 'Mumbai ' and 'Mumbai' are treated as different cities.")
    string_cols = cleaned_df.select_dtypes(include="object").columns.tolist()
    
    if string_cols:
        selected_str_col = st.selectbox("Select text column", string_cols)
        str_operations = st.multiselect("Select operations", [
            "Strip whitespace",
            "Convert to lowercase",
            "Convert to uppercase",
            "Remove special characters",
            "Remove extra spaces"
        ])
        
        if st.button("Apply String Cleaning"):
            if "Strip whitespace" in str_operations:
                cleaned_df[selected_str_col] = cleaned_df[selected_str_col].str.strip()
            if "Convert to lowercase" in str_operations:
                cleaned_df[selected_str_col] = cleaned_df[selected_str_col].str.lower()
            if "Convert to uppercase" in str_operations:
                cleaned_df[selected_str_col] = cleaned_df[selected_str_col].str.upper()
            if "Remove special characters" in str_operations:
                cleaned_df[selected_str_col] = cleaned_df[selected_str_col].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
            if "Remove extra spaces" in str_operations:
                cleaned_df[selected_str_col] = cleaned_df[selected_str_col].str.replace(r"\s+", " ", regex=True)
            st.success(f"✅ Done! Applied text cleaning to '{selected_str_col}'. Now values like ' Mumbai ' and 'Mumbai' will be treated as the same. This prevents wrong grouping in charts and analysis.")
            st.dataframe(cleaned_df[[selected_str_col]].head(10))
    else:
        st.info("No text columns found")

    # ── SECTION 6: Data Type Conversion ─────────────────────────
    st.markdown("## 🔄 Data Type Conversion")
    st.info("💡 Every column has a data type — numbers, text, or dates. If a number column is stored as text, you cannot do math on it. If a date column is stored as text, you cannot filter by date range. This section lets you fix those issues.")
    selected_convert_col = st.selectbox("Select column to convert", cleaned_df.columns.tolist())
    current_type = str(cleaned_df[selected_convert_col].dtype)
    st.info(f"Current type: **{current_type}**")
    
    target_type = st.selectbox("Convert to", ["int", "float", "string", "datetime"])
    
    if st.button("Convert Data Type"):
        try:
            if target_type == "int":
                cleaned_df[selected_convert_col] = pd.to_numeric(cleaned_df[selected_convert_col], errors="coerce").astype("Int64")
            elif target_type == "float":
                cleaned_df[selected_convert_col] = pd.to_numeric(cleaned_df[selected_convert_col], errors="coerce")
            elif target_type == "string":
                cleaned_df[selected_convert_col] = cleaned_df[selected_convert_col].astype(str)
            elif target_type == "datetime":
                cleaned_df[selected_convert_col] = pd.to_datetime(cleaned_df[selected_convert_col], errors="coerce")
            selected_col_name = selected_convert_col
            st.success(f"✅ Done! '{selected_col_name}' is now a {target_type} column. You can now use it properly in calculations and charts.")
            st.dataframe(cleaned_df.head(5))
        except Exception as e:
            st.error(f"Conversion failed: {e}")

    # ── SECTION 7: Outlier Detection ─────────────────────────────
    st.markdown("## 📉 Outlier Detection")
    st.info("💡 Outliers are extreme values that are very different from the rest — like a product priced at ₹1,00,00,000 when others are ₹500-₹5000. They can distort your charts and averages. The boxplot below shows dots outside the box — those are outliers.")
    numeric_cols = cleaned_df.select_dtypes(include="number").columns.tolist()
    
    if numeric_cols:
        outlier_col = st.selectbox("Select numeric column", numeric_cols)
        Q1 = cleaned_df[outlier_col].quantile(0.25)
        Q3 = cleaned_df[outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = cleaned_df[(cleaned_df[outlier_col] < Q1 - 1.5 * IQR) | 
                               (cleaned_df[outlier_col] > Q3 + 1.5 * IQR)]
        
        st.warning(f"Found {len(outliers)} outliers in '{outlier_col}'")
        
        fig = px.box(cleaned_df, y=outlier_col, title=f"Boxplot — {outlier_col}")
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Remove Outliers"):
            cleaned_df = cleaned_df[~((cleaned_df[outlier_col] < Q1 - 1.5 * IQR) | 
                                       (cleaned_df[outlier_col] > Q3 + 1.5 * IQR))]
            st.success(f"✅ Done! Removed {len(outliers)} extreme values from '{outlier_col}'. Your averages and charts will now be more accurate and representative of typical data.")

    # ── SECTION 8: Rename Columns ────────────────────────────────
    st.markdown("## ✏️ Rename Columns")
    st.info("💡 Column names like 'col1', 'rev_usd_q3' are confusing. Renaming them to clear names like 'Revenue' makes your analysis easier to read and share with others.")
    col_to_rename = st.selectbox("Select column to rename", cleaned_df.columns.tolist())
    new_name = st.text_input("New column name")
    if st.button("Rename Column") and new_name:
        cleaned_df = cleaned_df.rename(columns={col_to_rename: new_name})
        st.success(f"✅ Done! Column '{col_to_rename}' is now called '{new_name}'. This change only affects your current session — your original file is unchanged.")
        st.dataframe(cleaned_df.head(5))

    # ── SECTION 9: Export Cleaned Data ───────────────────────────
    st.markdown("## 💾 Export Cleaned Data")
    st.info("💡 Download your cleaned data as an Excel file. This is the final version after all the fixes you applied above — missing values filled, duplicates removed, text cleaned.")
    cleaned_excel = export_to_excel(cleaned_df)
    st.download_button(
        label="⬇️ Download Cleaned Data (Excel)",
        data=cleaned_excel,
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
