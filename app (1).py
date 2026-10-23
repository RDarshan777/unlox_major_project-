"""Interactive dashboard for Retail Analytics & AI-Powered Sales Forecasting System.

Run from the project folder after running the notebook:
    streamlit run app.py
"""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Retail Analytics Dashboard", page_icon="📊", layout="wide")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA_PATH = OUT / "cleaned_retail_data.csv"
FORECAST_PATH = OUT / "30_day_revenue_forecast.csv"
SEGMENTS_PATH = OUT / "store_segments.csv"

st.title("Retail Analytics & AI-Powered Sales Forecasting System")
st.caption("Interactive sales analysis, forecasts, store performance, and data-driven recommendations")

if not DATA_PATH.exists():
    st.error("Cleaned data was not found. Run all cells in major_project_R_Darshan_reviewed.ipynb first, then refresh this page.")
    st.stop()

@st.cache_data
def load_data(path_string):
    data = pd.read_csv(path_string)
    data.columns = data.columns.str.strip()
    required = {"Date", "Store_ID", "Revenue", "Units_Sold"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Cleaned data is missing required columns: {', '.join(sorted(missing))}")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Revenue"] = pd.to_numeric(data["Revenue"], errors="coerce")
    data["Units_Sold"] = pd.to_numeric(data["Units_Sold"], errors="coerce")
    data = data.dropna(subset=["Date", "Revenue", "Units_Sold"])
    return data

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load the cleaned dataset: {exc}")
    st.stop()

with st.sidebar:
    st.header("Filters")
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    stores = sorted(df["Store_ID"].dropna().astype(str).unique())
    chosen_stores = st.multiselect("Stores", stores, default=stores)
    if "Product_Category" in df.columns:
        categories = sorted(df["Product_Category"].dropna().astype(str).unique())
        chosen_categories = st.multiselect("Product categories", categories, default=categories)
    else:
        chosen_categories = []

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)
filtered = df[(df["Date"] >= start_date) & (df["Date"] <= end_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))]
filtered = filtered[filtered["Store_ID"].astype(str).isin(chosen_stores)]
if "Product_Category" in filtered.columns:
    filtered = filtered[filtered["Product_Category"].astype(str).isin(chosen_categories)]

if filtered.empty:
    st.warning("No rows match the selected filters. Change the filters in the sidebar.")
    st.stop()

revenue = filtered["Revenue"].sum()
units = filtered["Units_Sold"].sum()
store_count = filtered["Store_ID"].nunique()
avg_revenue = filtered.groupby("Date")["Revenue"].sum().mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total revenue", f"{revenue:,.2f}")
c2.metric("Units sold", f"{units:,.0f}")
c3.metric("Stores included", f"{store_count:,}")
c4.metric("Average daily revenue", f"{avg_revenue:,.2f}")

left, right = st.columns(2)
with left:
    daily = filtered.groupby("Date", as_index=False)["Revenue"].sum()
    st.subheader("Revenue over time")
    st.plotly_chart(px.line(daily, x="Date", y="Revenue", markers=True), use_container_width=True)
with right:
    st.subheader("Top stores by revenue")
    store_sales = filtered.groupby("Store_ID", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False).head(10)
    st.plotly_chart(px.bar(store_sales, x="Store_ID", y="Revenue", title="Top 10 stores"), use_container_width=True)

if "Product_Category" in filtered.columns:
    st.subheader("Revenue by product category")
    category_sales = filtered.groupby("Product_Category", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    st.plotly_chart(px.bar(category_sales, x="Product_Category", y="Revenue"), use_container_width=True)

st.subheader("30-day revenue forecast")
if FORECAST_PATH.exists():
    forecast = pd.read_csv(FORECAST_PATH)
    if {"Date", "Forecast_Revenue"}.issubset(forecast.columns):
        forecast["Date"] = pd.to_datetime(forecast["Date"], errors="coerce")
        forecast["Forecast_Revenue"] = pd.to_numeric(forecast["Forecast_Revenue"], errors="coerce")
        forecast = forecast.dropna(subset=["Date", "Forecast_Revenue"])
        if not forecast.empty:
            st.plotly_chart(px.line(forecast, x="Date", y="Forecast_Revenue", markers=True), use_container_width=True)
            st.caption("Forecasts are model estimates, not guarantees. The forecast is overall-system revenue and is not recalculated for the selected store/category filters.")
        else:
            st.info("Forecast file exists but has no usable forecast rows. Rerun the forecasting notebook cells.")
    else:
        st.info("Forecast file does not have the expected Date and Forecast_Revenue columns.")
else:
    st.info("Forecast file not found. Run the notebook forecasting and save cells first.")

st.subheader("Automated business insights")
insights = []
store_summary = filtered.groupby("Store_ID")["Revenue"].sum().sort_values(ascending=False)
if not store_summary.empty:
    insights.append(f"Highest-revenue store in the current selection: {store_summary.index[0]} ({store_summary.iloc[0]:,.2f}).")
    if len(store_summary) > 1:
        insights.append(f"Lowest-revenue store in the current selection: {store_summary.index[-1]} ({store_summary.iloc[-1]:,.2f}); review stock availability, location, and customer mix before taking action.")
if "Product_Category" in filtered.columns:
    category_summary = filtered.groupby("Product_Category")["Revenue"].sum().sort_values(ascending=False)
    if not category_summary.empty:
        share = category_summary.iloc[0] / revenue * 100 if revenue else 0
        insights.append(f"Top category: {category_summary.index[0]}, contributing {share:.1f}% of revenue in this selection. Check stock levels and profit margins before increasing inventory.")
monthly = filtered.set_index("Date")["Revenue"].resample("MS").sum()
if len(monthly) >= 2:
    changes = monthly.pct_change().replace([float("inf"), -float("inf")], pd.NA).dropna()
    if not changes.empty:
        best, worst = changes.idxmax(), changes.idxmin()
        insights.append(f"Largest monthly increase in this selection: {best.strftime('%B %Y')} ({changes.loc[best] * 100:.1f}%). Largest decrease: {worst.strftime('%B %Y')} ({changes.loc[worst] * 100:.1f}%). Investigate promotions, holidays, and availability; these patterns alone do not establish causes.")
for item in insights:
    st.markdown(f"- {item}")

if SEGMENTS_PATH.exists():
    st.subheader("Store segmentation")
    segments = pd.read_csv(SEGMENTS_PATH)
    if "Store_ID" in segments.columns and "Segment" in segments.columns:
        st.dataframe(segments, use_container_width=True, hide_index=True)
    else:
        st.caption("Store segment file exists, but the expected columns were not found.")

with st.expander("View filtered data"):
    st.dataframe(filtered, use_container_width=True, hide_index=True)

st.caption("Recommendations are rule-based summaries of observed data, not proof of causation or a substitute for business review.")
