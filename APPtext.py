import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Financial Statement Analyzer", layout="wide")

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
def get_ticker(t):
    s = yf.Ticker(t)
    return s.info, s.income_stmt, s.balance_sheet, s.cashflow

@st.cache_data(ttl=3600)
def get_history(t, period="5y"):
    return yf.Ticker(t).history(period=period)

def fmt(val):
    if pd.isna(val): return "N/A"
    v = float(val)
    for threshold, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(v) >= threshold:
            return f"${v / threshold:.2f}{suffix}"
    return f"${v:.2f}"

def sget(df, key):
    try:
        return float(df.loc[key].iloc[0]) if key in df.index else np.nan
    except (IndexError, TypeError, ValueError):
        return np.nan

def sdiv(a, b):
    return a / b if not pd.isna(a) and not pd.isna(b) and b != 0 else np.nan

def years_data(df, rows):
    result, yrs = {}, []
    for col in reversed(list(df.columns)):
        yr = col.strftime("%Y") if hasattr(col, "strftime") else str(col)
        yrs.append(yr)
        result[yr] = {r: float(df.loc[r, col]) if r in df.index and not pd.isna(df.loc[r, col]) else 0 for r in rows}
    return yrs, result

def grouped_bar(x, traces, height=400):
    fig = go.Figure([go.Bar(x=x, y=vals, name=name, marker_color=color) for name, vals, color in traces])
    fig.update_layout(barmode="group", height=height, yaxis_title="Amount ($)")
    st.plotly_chart(fig, use_container_width=True)

def line_chart(x, traces, height=400, ytitle=""):
    fig = go.Figure([go.Scatter(x=x, y=vals, mode="lines+markers", name=name) for name, vals in traces])
    fig.update_layout(height=height, yaxis_title=ytitle)
    st.plotly_chart(fig, use_container_width=True)

def show_table(df, rows):
    data = {}
    for col in df.columns:
        yr = col.strftime("%Y") if hasattr(col, "strftime") else str(col)
        data[yr] = {r: fmt(df.loc[r, col]) if r in df.index and not pd.isna(df.loc[r, col]) else "N/A" for r in rows}
    st.dataframe(pd.DataFrame(data), use_container_width=True)

def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, val) in zip(cols, items):
        col.metric(label, val)

def pct(v):
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) and v is not None else "N/A"

def num(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) and v is not None else "N/A"

# ── Sidebar ──
st.sidebar.title("Financial Analyzer")
selected = st.sidebar.selectbox("Select Company", list(COMPANIES.keys()))
ticker = COMPANIES[selected]
custom = st.sidebar.text_input("Or enter a custom ticker:")
if custom.strip():
    ticker = custom.strip().upper()

page = st.sidebar.radio("Analysis Type", [
    "Overview", "Income Statement", "Balance Sheet", "Cash Flow",
    "Financial Ratios", "Stock Price Analysis", "Company Comparison",
])

st.title("Financial Statement Analyzer")
st.caption(f"Analyzing: **{ticker}**")

try:
    info, income, bs, cf = get_ticker(ticker)

    if page == "Overview":
        pe = info.get("trailingPE")
        dy = info.get("dividendYield")
        metric_row([
            ("Market Cap", fmt(info.get("marketCap", 0))), 
            ("Revenue (TTM)", fmt(info.get("totalRevenue", 0))),
            ("Net Income", fmt(info.get("netIncomeToCommon", 0))), 
            ("EPS", f"${info.get('trailingEps', 'N/A')}"),
        ])
        metric_row([
            ("P/E Ratio", num(pe)), 
            ("Dividend Yield", pct(dy) if dy else "N/A"),
            ("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}"), 
            ("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}"),
        ])
        st.subheader("Company Profile")
        st.write(info.get("longBusinessSummary", "No description available."))
        hist = get_history(ticker, "1y")
        if not hist.empty:
            fig = px.line(hist, y="Close", title=f"{ticker} - 1 Year Stock Price")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Income Statement" and income is not None and not income.empty:
        rows = ["Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Expense",
                "Operating Income", "Net Income", "EBITDA", "Basic EPS", "Diluted EPS"]
        st.subheader("Income Statement")
        show_table(income, rows)
        yrs, d = years_data(income, ["Total Revenue", "Net Income", "Gross Profit", "Operating Income"])
        st.subheader("Revenue vs Net Income Trend")
        grouped_bar(yrs, [
            ("Revenue", [d[y]["Total Revenue"] for y in yrs], "#2563eb"),
            ("Net Income", [d[y]["Net Income"] for y in yrs], "#16a34a")
        ])
        st.subheader("Profit Margins Over Time")
        line_chart(yrs, [
            ("Gross Margin %", [sdiv(d[y]["Gross Profit"], d[y]["Total Revenue"]) * 100 for y in yrs]),
            ("Operating Margin %", [sdiv(d[y]["Operating Income"], d[y]["Total Revenue"]) * 100 for y in yrs]),
            ("Net Margin %", [sdiv(d[y]["Net Income"], d[y]["Total Revenue"]) * 100 for y in yrs]),
        ], ytitle="Margin (%)")

    elif page == "Balance Sheet" and bs is not None and not bs.empty:
        rows = ["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity",
                "Current Assets", "Current Liabilities", "Cash And Cash Equivalents",
                "Total Debt", "Net Tangible Assets", "Working Capital"]
        st.subheader("Balance Sheet")
        show_table(bs, rows)
        yrs, d = years_data(bs, ["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity"])
        st.subheader("Assets vs Liabilities vs Equity")
        grouped_bar(yrs, [
            ("Total Assets", [d[y]["Total Assets"] for y in yrs], "#2563eb"),
            ("Total Liabilities", [d[y]["Total Liabilities Net Minority Interest"] for y in yrs], "#dc2626"),
            ("Equity", [d[y]["Stockholders Equity"] for y in yrs], "#16a34a"),
        ])
        st.subheader("Debt to Equity Ratio Over Time")
        line_chart(yrs, [("D/E Ratio", [sdiv(d[y]["Total Liabilities Net Minority Interest"], d[y]["Stockholders Equity"]) for y in yrs])], 350, "Ratio")

    elif page == "Cash Flow" and cf is not None and not cf.empty:
        rows = ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow",
                "Free Cash Flow", "Capital Expenditure", "Depreciation And Amortization"]
        st.subheader("Cash Flow Statement")
        show_table(cf, rows)
        yrs, d = years_data(cf, ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Free Cash Flow"])
        st.subheader("Cash Flow Breakdown")
        grouped_bar(yrs, [
            ("Operating", [d[y]["Operating Cash Flow"] for y in yrs], "#2563eb"),
            ("Investing", [d[y]["Investing Cash Flow"] for y in yrs], "#dc2626"),
            ("Financing", [d[y]["Financing Cash Flow"] for y in yrs], "#f59e0b"),
        ])
        st.subheader("Free Cash Flow Trend")
        fcf = [d[y]["Free Cash Flow"] for y in yrs]
        fig = go.Figure(go.Bar(x=yrs, y=fcf, marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in fcf]))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Financial Ratios":
        st.subheader("Key Financial Ratios")
        ebitda_m = sdiv(info.get("ebitda", np.nan), info.get("totalRevenue", np.nan))
        sections = [
            ("Profitability", [
                ("Gross Margin", pct(info.get("grossMargins"))), 
                ("Operating Margin", pct(info.get("operatingMargins"))),
                ("Net Profit Margin", pct(info.get("profitMargins"))),
            ]),
            ("Profitability (cont.)", [
                ("ROE", pct(info.get("returnOnEquity"))), 
                ("ROA", pct(info.get("returnOnAssets"))),
                ("EBITDA Margin", pct(ebitda_m) if not pd.isna(ebitda_m) else "N/A"),
            ]),
            ("Liquidity", [
                ("Current Ratio", num(info.get("currentRatio"))), 
                ("Quick Ratio", num(info.get("quickRatio"))),
                ("Cash Ratio", num(sdiv(sget(bs, "Cash And Cash Equivalents"), sget(bs, "Current Liabilities"))) if bs is not None else "N/A"),
            ]),
            ("Leverage", [
                ("Debt to Equity", num(info.get("debtToEquity", 0) / 100) if info.get("debtToEquity") else "N/A"),
                ("Debt to Assets", num(sdiv(sget(bs, "Total Debt"), sget(bs, "Total Assets"))) if bs is not None else "N/A"),
                ("Equity Multiplier", num(sdiv(sget(bs, "Total Assets"), sget(bs, "Stockholders Equity"))) if bs is not None else "N/A"),
            ]),
            ("Valuation", [
                ("P/E Ratio", num(info.get("trailingPE"))), 
                ("P/B Ratio", num(info.get("priceToBook"))),
                ("P/S Ratio", num(info.get("priceToSalesTrailing12Months"))),
            ])
        ]
        for section, items in sections:
            st.markdown(f"### {section}")
            metric_row(items)

    elif page == "Stock Price Analysis":
        st.subheader("Stock Price Analysis & Forecasting")
        period = st.selectbox("Time Period", ["1y", "2y", "5y", "10y", "max"], index=2)
        hist = get_history(ticker, period)
        if not hist.empty:
            fig = go.Figure(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"]))
            fig.update_layout(title=f"{ticker} Stock Price", height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast Logic
            st.subheader("Price Forecast (Linear Regression)")
            days = st.slider("Forecast Days", 7, 90, 30)
            X = np.arange(len(hist)).reshape(-1, 1)
            model = LinearRegression().fit(X, hist["Close"].values)
            future_X = np.arange(len(hist), len(hist) + days).reshape(-1, 1)
            future_dates = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=days, freq="B")
            
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Historical"))
            fig_fc.add_trace(go.Scatter(x=future_dates, y=model.predict(future_X), name="Forecast", line=dict(dash='dash', color='#dc2626')))
            st.plotly_chart(fig_fc, use_container_width=True)

    elif page == "Company Comparison":
        st.subheader("Compare Multiple Companies")
        selected_names = st.multiselect("Select companies", list(COMPANIES.keys()), default=list(COMPANIES.keys())[:3])
        if len(selected_names) >= 2:
            tickers = [COMPANIES[n] for n in selected_names]
            comp_data = {t: get_ticker(t)[0].get("marketCap", 0) for t in tickers}
            st.bar_chart(pd.Series(comp_data))

except Exception as e:
    st.error(f"Error fetching data: {e}")

st.divider()
st.caption("Data from Yahoo Finance. For educational purposes only - not financial advice.")
