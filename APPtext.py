import streamlit as st
import asyncio
import pandas as pd
import tempfile
import os
import requests
import time
import json
import re
import google.generativeai as genai
import plotly.express as px
from pydantic import BaseModel

# --- 1. Universal Financial Data Model ---
class UniversalFinancials(BaseModel):
    company_name: str
    fiscal_year: int
    industry_category: str
    total_revenue: float
    operating_expenses: float
    net_income: float
    total_assets: float
    total_liabilities: float
    total_equity: float
    total_debt: float
    operating_cash_flow: float

# --- 2. Smart Filtering ---
def extract_financial_sections(raw_md: str, window_size: int = 30000) -> str:
    keywords = ["CONSOLIDATED BALANCE SHEET", "PROFIT AND LOSS", "CASH FLOW STATEMENT", "FINANCIAL STATEMENTS"]
    raw_md_upper = raw_md.upper()
    start_index = -1
    for kw in keywords:
        idx = raw_md_upper.find(kw)
        if idx != -1:
            start_index = max(0, idx - 500)
            break
    return raw_md[start_index : start_index + window_size] if start_index != -1 else raw_md[:window_size]

# --- 3. Cached API Calls with Unit Normalization ---
@st.cache_data(show_spinner="AI Analysis...")
def normalize_with_llm_cached(raw_md, gemini_key):
    relevant_text = extract_financial_sections(raw_md)
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = f"""
    You are a Senior Financial Controller. Extract metrics from the report provided.
    
    CRITICAL INSTRUCTIONS:
    1. CONVERT ALL VALUES TO CRORES (INR). 
       - If the report is in Lakhs, divide by 100.
       - If the report is in Millions/Billions, convert accordingly.
    2. FISCAL_YEAR: Use ONLY a 4-digit integer (e.g., 2024). No "FY" or "2023-24".
    3. RULES: If Bank, Revenue = Interest + Other Income. If Corporate, Revenue = Sales.
    
    Return ONLY a JSON object:
    {{
    "company_name": "String", "fiscal_year": 2024, "industry_category": "String",
    "total_revenue": 0.0, "operating_expenses": 0.0, "net_income": 0.0,
    "total_assets": 0.0, "total_liabilities": 0.0, "total_equity": 0.0,
    "total_debt": 0.0, "operating_cash_flow": 0.0
    }}
    
    Data: {relevant_text}
    """
    res = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    return res.text

# --- 4. The "Safe" Pipeline (Fixes Validation Errors) ---
def safe_validate_financials(json_str: str) -> dict:
    """Cleans up LLM output before Pydantic sees it."""
    raw_dict = json.loads(json_str)
    
    # Fix Fiscal Year if it's a string like "FY 2023-24"
    if isinstance(raw_dict.get("fiscal_year"), str):
        match = re.search(r'(\d{4})', raw_dict["fiscal_year"])
        raw_dict["fiscal_year"] = int(match.group(1)) if match else 0
        
    # Standardize field names just in case
    data = UniversalFinancials(**raw_dict)
    
    # Calculate Ratios
    m = data.model_dump()
    m["net_margin_%"] = (data.net_income / data.total_revenue * 100) if data.total_revenue else 0
    m["roe_%"] = (data.net_income / data.total_equity * 100) if data.total_equity else 0
    m["debt_equity"] = (data.total_debt / data.total_equity) if data.total_equity else 0
    return m

async def run_pipeline(files):
    results = []
    # Sequential processing to prevent 429 errors
    for f in files:
        # Step 1: Parse (Using your existing parse_pdf_cached logic)
        raw_md = await asyncio.to_thread(parse_pdf_cached, f.getvalue(), f.name, st.secrets["LLAMA_CLOUD_KEY"])
        # Step 2: Normalize
        json_str = normalize_with_llm_cached(raw_md, st.secrets["GEMINI_API_KEY"])
        # Step 3: Safe Validate
        results.append(safe_validate_financials(json_str))
    return results

# --- 5. UI Layout ---
st.set_page_config(page_title="Financial Analyzer", layout="wide")
st.title("📈 Universal Corporate Financial Analyzer")
uploaded = st.file_uploader("Upload Annual Reports (PDF)", type="pdf", accept_multiple_files=True)

if st.button("Run Analysis") and uploaded:
    try:
        data_list = asyncio.run(run_pipeline(uploaded))
        df = pd.DataFrame(data_list).sort_values("fiscal_year")
        
        st.success("Data successfully normalized to INR Crores.")
        st.dataframe(df.style.format(precision=2))
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(df, x="fiscal_year", y=["total_revenue", "net_income"], barmode="group", title="Revenue & Profit (in Crores)"), use_container_width=True)
        with c2:
            st.plotly_chart(px.line(df, x="fiscal_year", y=["net_margin_%", "roe_%"], markers=True, title="Efficiency Margins (%)"), use_container_width=True)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
