import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import io

# --- 1. CONFIGURATION & HELPERS ---
st.set_page_config(page_title="Pro Financial Analyzer", layout="wide")

COMPANIES = {
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS", 
    "TCS (TCS.NS)": "TCS.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS", 
    "Infosys (INFY.NS)": "INFY.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS", 
    "Bharti Airtel (BHARTIARTL.NS)": "BHARTIARTL.NS",
    "SBI (SBIN.NS)": "SBIN.NS", 
    "ITC (ITC.NS)": "ITC.NS",
    "Wipro (WIPRO.NS)": "WIPRO.NS", 
    "L&T (LT.NS)": "LT.NS",
}

@st.cache_data(ttl=3600)
def get_ticker_data(t):
    s = yf.Ticker(t)
    return s.info, s.income_stmt, s.balance_sheet, s.cashflow

@st.cache_data(ttl=3600)
def get_history(t, period="5y"):
    return yf.Ticker(t).history(period=period)

def fmt_inr(val):
    if pd.isna(val) or val == 0: return "N/A"
    v = float(val)
    # 1e7 = 1 Crore, 1e5 = 1 Lakh
    for threshold, suffix in [(1e12, "T"), (1e9, "B"), (1e7, " Cr"), (1e5, " L"), (1e3, " K")]:
        if abs(v) >= threshold:
            return f"₹{v / threshold:.2f}{suffix}"
    return f"₹{v:.2f}"

def sdiv(a, b):
    return a / b if not pd.isna(a) and not pd.isna(b) and b != 0 else np.nan

def pct(v):
    return f"{v * 100:.2f}%" if isinstance(v, (int, float)) and not pd.isna(v) else "N/A"

def num(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) and not pd.isna(v) else "N/A"

# --- Feature 1: Fundamental "Red Flag" Analysis ---
def run_health_check(income, bs, cf):
    flags = []
    if income is not None and not income.empty:
        if "Net Income" in income.index and income.shape[1] > 1:
            if income.loc["Net Income"].iloc[0] < income.loc["Net Income"].iloc[1]:
                flags.append("🚨 **Profitability:** Net Income has declined year-over-year.")
        
        if "Total Revenue" in income.index and "Net Income" in income.index:
            margin_now = income.loc["Net Income"].iloc[0] / income.loc["Total Revenue"].iloc[0]
            margin_prev = income.loc["Net Income"].iloc[1] / income.loc["Total Revenue"].iloc[1]
            if margin_now < margin_prev:
                flags.append("⚠️ **Efficiency:** Net Profit Margin is shrinking.")

    if bs is not None and not bs.empty:
        if "Total Debt" in bs.index and bs.shape[1] > 1:
            if bs.loc["Total Debt"].iloc[0] > bs.loc["Total Debt"].iloc[1] * 1.15:
                flags.append("🚨 **Solvency:** Total Debt increased by >15% this year.")

    if cf is not None and not cf.empty:
        if "Operating Cash Flow" in cf.index:
            if cf.loc["Operating Cash Flow"].iloc[0] < 0:
                flags.append("🚨 **Cash Flow:** Operating Cash Flow is Negative.")
    
    return flags if flags else ["✅ **Fundamental Health:** No major red flags detected."]

# --- 2. SIDEBAR ---
st.sidebar.title("Financial Analyzer Pro")
selected = st.sidebar.selectbox("Select Company", list(COMPANIES.keys()))
ticker = COMPANIES[selected]
custom = st.sidebar.text_input("Or enter custom ticker (e.g., RELIANCE.NS):")
if custom.strip(): ticker = custom.strip().upper()

page = st.sidebar.radio("Analysis Type", [
    "Dashboard & Red Flags", "Income Statement", "Balance Sheet", "Cash Flow",
    "Intrinsic Value (DCF)", "Peer Comparison", "Stock Price Analysis"
])

# --- 3. MAIN UI LOGIC ---
st.title("Financial Statement Analyzer")
st.caption(f"Analyzing: **{ticker}** | Values in **INR**")

try:
    info, income, bs, cf = get_ticker_data(ticker)

    if page == "Dashboard & Red Flags":
        cols = st.columns(4)
        cols[0].metric("Market Cap", fmt_inr(info.get("marketCap", 0)))
        cols[1].metric("P/E Ratio", num(info.get("trailingPE")))
        cols[2].metric("ROE", pct(info.get("returnOnEquity")))
        cols[3].metric("Current Price", f"₹{info.get('currentPrice', 'N/A')}")
        
        st.subheader("🚩 Fundamental Health Check")
        for alert in run_health_check(income, bs, cf):
            st.write(alert)
        
        st.subheader("Company Profile")
        st.write(info.get("longBusinessSummary", "No description available."))

    elif page in ["Income Statement", "Balance Sheet", "Cash Flow"]:
        df_map = {"Income Statement": income, "Balance Sheet": bs, "Cash Flow": cf}
        df = df_map[page]
        st.subheader(f"{page} Data")
        st.dataframe(df.style.format(precision=2))
        
        # Feature: CSV Export
        csv = df.to_csv().encode('utf-8')
        st.download_button(f"📥 Download {page} as CSV", csv, f"{ticker}_{page.lower().replace(' ', '_')}.csv", "text/csv")

    elif page == "Intrinsic Value (DCF)":
        st.subheader("Simplified Discounted Cash Flow (DCF) Model")
        [Image of the discounted cash flow formula]
        fcf = cf.loc["Free Cash Flow"].iloc[0] if cf is not None and "Free Cash Flow" in cf.index else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Latest Free Cash Flow
