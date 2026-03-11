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
    "Bharti Airtel (BHARTIARTL.NS)": "BHARTIARTL.NS",
    "SBI (SBIN.NS)": "SBIN.NS", 
    "ITC (ITC.NS)": "ITC.NS",
    "Wipro (WIPRO.NS)": "WIPRO.NS", 
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
    # 1e7 = 1 Crore, 1e5 = 1 Lakh
    for threshold, suffix in [(1e7, " Cr"), (1e5, " L"), (1e3, " K")]:
        if abs(v) >= threshold:
            return f"₹{v / threshold:.2f}{suffix}"
    return f"₹{v:.2f}"

def sdiv(a, b):
    return a / b if not pd.isna(a) and not pd.isna(b) and b != 0 else np.nan

def years_data(df, rows):
    result, yrs = {}, []
    if df is None or df.empty: return [], {}
    for col in reversed(list(df.columns)):
        yr = col.strftime("%Y") if hasattr(col, "strftime") else str(col)
        yrs.append(yr)
        result[yr] = {r: float(df.loc[r, col]) if r in df.index and not pd.isna(df.loc[r, col]) else 0 for r in rows}
    return yrs, result

# --- UI Components ---
def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, val) in zip(cols, items):
        col.metric(label, val)

def grouped_bar(x, traces, height=400):
    fig = go.Figure([go.Bar(x=x, y=vals, name=name, marker_color=color) for name, vals, color in traces])
    fig.update_layout(barmode="group", height=height, yaxis_title="Amount (₹)")
    st.plotly_chart(fig, use_container_width=True)

def line_chart(x, traces, height=400, ytitle=""):
    fig = go.Figure([go.Scatter(x=x, y=vals, mode="lines+markers", name=name) for name, vals in traces])
    fig.update_layout(height=height, yaxis_title=ytitle)
    st.plotly_chart(fig, use_container_width=True)

def show_table(df, rows):
    data = {}
    for col in df.columns:
        yr = col.strftime("%Y") if hasattr(col, "strftime") else str(col)
        data[yr] = {r: fmt_inr(df.loc[r, col]) if r in df.index and not pd.isna(df.loc[r, col]) else "N/A" for r in rows}
    st.dataframe(pd.DataFrame(data), use_container_width=True)

# --- Sidebar ---
st.sidebar.title("Financial Analyzer")
selected = st.sidebar.selectbox("Select Company", list(COMPANIES.keys()))
ticker = COMPANIES[selected]
custom = st.sidebar.text_input("Or enter a custom ticker (e.g., RELIANCE.NS):")
if custom.strip():
    ticker = custom.strip().upper()

page = st.sidebar.radio("Analysis Type", [
    "Overview", "Income Statement", "Balance Sheet", "Cash Flow",
    "Financial Ratios", "Stock Price Analysis"
])

st.title("Financial Statement Analyzer")
st.caption(f"Analyzing: **{ticker}** | Values in **INR Crores/Lakhs**")

try:
    info, income, bs, cf = get_ticker_data(ticker)

    if page == "Overview":
        metric_row([
            ("Market Cap", fmt_inr(info.get("marketCap", 0))), 
            ("Revenue (TTM)", fmt_inr(info.get("totalRevenue", 0))),
            ("Net Income", fmt_inr(info.get("netIncomeToCommon", 0))), 
            ("EPS", f"₹{info.get('trailingEps', 'N/A')}"),
        ])
        metric_row([
            ("P/E Ratio", f"{info.get('trailingPE', 'N/A')}"), 
            ("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A"),
            ("52W High", f"₹{info.get('fiftyTwoWeekHigh', 'N/A')}"), 
            ("52W Low", f"₹{info.get('fiftyTwoWeekLow', 'N/A')}"),
        ])
        st.subheader("Company Profile")
        st.write(info.get("longBusinessSummary", "No description available."))
        hist = get_history(ticker, "1y")
        if not hist.empty:
            fig = px.line(hist, y="Close", title=f"{ticker} - 1 Year Price Action", color_discrete_sequence=["#16a34a"])
            fig.update_layout(height=400, yaxis_title="Price (₹)")
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Income Statement" and income is not None:
        rows = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "EBITDA"]
        st.subheader("Income Statement")
        show_table(income, rows)
        yrs, d = years_data(income, rows)
        if yrs:
            grouped_bar(yrs, [
                ("Revenue", [d[y]["Total Revenue"] for y in yrs], "#2563eb"),
                ("Net Profit", [d[y]["Net Income"] for y in yrs], "#16a34a")
            ])

    elif page == "Balance Sheet" and bs is not None:
        rows = ["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity", "Total Debt"]
        st.subheader("Balance Sheet")
        show_table(bs, rows)
        yrs, d = years_data(bs, rows)
        if yrs:
            grouped_bar(yrs, [
                ("Assets", [d[y]["Total Assets"] for y in yrs], "#2563eb"),
                ("Liabilities", [d[y]["Total Liabilities Net Minority Interest"] for y in yrs], "#dc2626"),
                ("Equity", [d[y]["Stockholders Equity"] for y in yrs], "#16a34a")
            ])

    elif page == "Cash Flow" and cf is not None:
        rows = ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Free Cash Flow"]
        st.subheader("Cash Flow Statement")
        show_table(cf, rows)
        yrs, d = years_data(cf, rows)
        if yrs:
            grouped_bar(yrs, [
                ("Operating", [d[y]["Operating Cash Flow"] for y in yrs], "#2563eb"),
                ("Free Cash Flow", [d[y]["Free Cash Flow"] for y in yrs], "#16a34a")
            ])

    elif page == "Financial Ratios":
        st.subheader("Key Ratios")
        metric_row([
            ("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%"),
            ("ROA", f"{info.get('returnOnAssets', 0)*100:.2f}%"),
            ("Net Margin", f"{info.get('profitMargins', 0)*100:.2f}%")
        ])
        metric_row([
            ("Current Ratio", f"{info.get('currentRatio', 'N/A')}"),
            ("Debt to Equity", f"{info.get('debtToEquity', 0)/100:.2f}" if info.get('debtToEquity') else "N/A"),
            ("P/B Ratio", f"{info.get('priceToBook', 'N/A')}")
        ])

    elif page == "Stock Price Analysis":
        period = st.selectbox("Time Period", ["1y", "2y", "5y", "max"], index=0)
        hist = get_history(ticker, period)
        if not hist.empty:
            fig = go.Figure(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"]))
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, yaxis_title="Price (₹)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast
            st.subheader("30-Day Forecast (Linear Trend)")
            X = np.arange(len(hist)).reshape(-1, 1)
            model = LinearRegression().fit(X, hist["Close"].values)
            future_X = np.arange(len(hist), len(hist) + 30).reshape(-1, 1)
            future_dates = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=30, freq="B")
            
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="History"))
            fig_fc.add_trace(go.Scatter(x=future_dates, y=model.predict(future_X), name="Forecast", line=dict(dash='dash', color='#dc2626')))
            st.plotly_chart(fig_fc, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")

st.divider()
st.caption("Data: Yahoo Finance | For educational use only.")
