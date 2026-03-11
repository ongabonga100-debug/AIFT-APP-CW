import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression

# --- Page Configuration ---
st.set_page_config(page_title="Indian Financial Analyzer", layout="wide")

# --- Default Companies ---
COMPANIES = {
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS", 
    "TCS (TCS.NS)": "TCS.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS", 
    "Infosys (INFY.NS)": "INFY.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS", 
    "SBI (SBIN.NS)": "SBIN.NS", 
    "ITC (ITC.NS)": "ITC.NS",
    "L&T (LT.NS)": "LT.NS",
}

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def get_ticker_data(t):
    s = yf.Ticker(t)
    return s.info, s.income_stmt, s.balance_sheet, s.cashflow

@st.cache_data(ttl=3600)
def get_history(t, period="5y"):
    return yf.Ticker(t).history(period=period)

def fmt_inr(val):
    """Formats values into Indian Numbering System (Crores/Lakhs)."""
    if pd.isna(val) or val == 0: return "N/A"
    v = float(val)
    # yfinance usually returns absolute INR. 
    # 1e7 = 1 Crore, 1e5 = 1 Lakh
    if abs(v) >= 1e7:
        return f"₹{v / 1e7:.2f} Cr"
    elif abs(v) >= 1e5:
        return f"₹{v / 1e5:.2f} L"
    elif abs(v) >= 1e3:
        return f"₹{v / 1e3:.2f} K"
    return f"₹{v:.2f}"

def years_data(df, rows):
    result, yrs = {}, []
    if df is None or df.empty: return [], {}
    for col in reversed(list(df.columns)):
        yr = col.strftime("%Y") if hasattr(col, "strftime") else str(col)
        yrs.append(yr)
        # We store raw values here for calculations, but format them for display
        result[yr] = {r: float(df.loc[r, col]) if r in df.index and not pd.isna(df.loc[r, col]) else 0 for r in rows}
    return yrs, result

# --- UI Components ---
def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, val) in zip(cols, items):
        col.metric(label, val)

def grouped_bar(x, traces, height=400):
    fig = go.Figure([go.Bar(x=x, y=vals, name=name) for name, vals, color in traces])
    fig.update_layout(barmode="group", height=height, yaxis_title="Value in INR", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def show_table(df, rows):
    data = {}
    for col in df.columns:
        yr = col.strftime("%Y") if hasattr(col, "strftime") else str(col)
        data[yr] = {r: fmt_inr(df.loc[r, col]) if r in df.index else "N/A" for r in rows}
    st.dataframe(pd.DataFrame(data), use_container_width=True)

# --- Sidebar ---
st.sidebar.title("🇮🇳 Finance Analyzer")
selected = st.sidebar.selectbox("Select Company", list(COMPANIES.keys()))
ticker = COMPANIES[selected]
custom = st.sidebar.text_input("Or Enter Custom Ticker (e.g. TATAMOTORS.NS):")
if custom.strip():
    ticker = custom.strip().upper()

page = st.sidebar.radio("Analysis Type", ["Overview", "Income Statement", "Balance Sheet", "Cash Flow", "Financial Ratios"])

st.title("Financial Statement Analyzer")
st.caption(f"Company: **{ticker}** | Currency: **INR** | Format: **Indian (Cr/L)**")

try:
    info, income, bs, cf = get_ticker_data(ticker)

    if page == "Overview":
        metric_row([
            ("Market Cap", fmt_inr(info.get("marketCap", 0))), 
            ("Revenue (TTM)", fmt_inr(info.get("totalRevenue", 0))),
            ("Net Profit (TTM)", fmt_inr(info.get("netIncomeToCommon", 0))), 
            ("Current Price", f"₹{info.get('currentPrice', 'N/A')}"),
        ])
        st.subheader("Business Summary")
        st.write(info.get("longBusinessSummary", "Summary not available."))
        
        hist = get_history(ticker, "1y")
        if not hist.empty:
            fig = px.line(hist, y="Close", title="1 Year Price Trend", labels={"Close": "Price (₹)"})
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Income Statement" and income is not None:
        rows = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "EBITDA"]
        show_table(income, rows)
        yrs, d = years_data(income, rows)
        if yrs:
            # We plot in Crores to keep the Y-axis readable
            grouped_bar(yrs, [
                ("Revenue (Cr)", [d[y]["Total Revenue"]/1e7 for y in yrs], "#2563eb"),
                ("Net Profit (Cr)", [d[y]["Net Income"]/1e7 for y in yrs], "#16a34a")
            ])

    elif page == "Balance Sheet" and bs is not None:
        rows = ["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity", "Total Debt"]
        show_table(bs, rows)
        yrs, d = years_data(bs, rows)
        if yrs:
            grouped_bar(yrs, [
                ("Assets (Cr)", [d[y]["Total Assets"]/1e7 for y in yrs], "#2563eb"),
                ("Debt (Cr)", [d[y]["Total Debt"]/1e7 for y in yrs], "#dc2626")
            ])

    elif page == "Cash Flow" and cf is not None:
        rows = ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Free Cash Flow"]
        show_table(cf, rows)
        yrs, d = years_data(cf, rows)
        if yrs:
            grouped_bar(yrs, [
                ("Operating CF (Cr)", [d[y]["Operating Cash Flow"]/1e7 for y in yrs], "#16a34a"),
                ("Free CF (Cr)", [d[y]["Free Cash Flow"]/1e7 for y in yrs], "#2563eb")
            ])

    elif page == "Financial Ratios":
        st.subheader("Efficiency & Valuation")
        metric_row([
            ("ROE (%)", f"{info.get('returnOnEquity', 0)*100:.2f}%"),
            ("Net Margin (%)", f"{info.get('profitMargins', 0)*100:.2f}%"),
            ("Debt/Equity", f"{info.get('debtToEquity', 0)/100:.2f}")
        ])
        metric_row([
            ("P/E Ratio", f"{info.get('trailingPE', 'N/A')}"),
            ("P/B Ratio", f"{info.get('priceToBook', 'N/A')}"),
            ("EV/EBITDA", f"{info.get('enterpriseToEbitda', 'N/A')}")
        ])

except Exception as e:
    st.error(f"Data Fetching Error: {e}")

st.divider()
st.caption("Disclaimer: Data is sourced from Yahoo Finance. Values are converted to the Indian numbering system for better readability.")
