import streamlit as st
import asyncio
import pandas as pd
import tempfile
import os
import requests
import time
import google.generativeai as genai
import plotly.express as px
from pydantic import BaseModel

# --- Fetch Keys Securely ---
LLAMA_CLOUD_KEY = st.secrets["LLAMA_CLOUD_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

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

# --- 2. NEW: Intelligent Section Filter ---
def extract_financial_sections(raw_md: str, window_size: int = 25000) -> str:
    """Isolates financial tables to save tokens and avoid quota errors."""
    keywords = ["CONSOLIDATED BALANCE SHEET", "PROFIT AND LOSS", "CASH FLOW STATEMENT", "FINANCIAL STATEMENTS"]
    raw_md_upper = raw_md.upper()
    start_index = -1
    for kw in keywords:
        idx = raw_md_upper.find(kw)
        if idx != -1:
            start_index = max(0, idx - 500)
            break
    return raw_md[start_index : start_index + window_size] if start_index != -1 else raw_md[:window_size]

# --- 3. UPDATED: Cached LlamaParse & Gemini Logic ---
@st.cache_data(show_spinner="Parsing PDF (Results Cached)...")
def parse_pdf_cached(file_bytes: bytes, file_name: str, llama_key: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    headers = {"accept": "application/json", "Authorization": f"Bearer {llama_key}"}
    try:
        with open(tmp_path, "rb") as f:
            files = {"file": (file_name, f, "application/pdf")}
            upload_res = requests.post("https://api.cloud.llamaindex.ai/api/parsing/upload", headers=headers, files=files)
            upload_res.raise_for_status()
            job_id = upload_res.json()["id"]
            while True:
                time.sleep(2)
                status = requests.get(f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}", headers=headers).json()["status"]
                if status == "SUCCESS":
                    return requests.get(f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/markdown", headers=headers).json()["markdown"]
                elif status == "ERROR":
                    raise Exception(f"LlamaParse failed for {file_name}")
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

@st.cache_data(show_spinner="Running AI Normalization...")
def normalize_with_llm_cached(raw_markdown: str, gemini_key: str) -> str:
    relevant_text = extract_financial_sections(raw_markdown)
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    You are an expert financial analyst. Extract metrics into JSON from the following data.
    If Bank: Revenue = Interest + Other Income. If Corporate: Revenue = Sales.
    Report: {relevant_text}
    """
    res = model.generate_content(prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0.1))
    return res.text

# --- 4. Logic & Calculations ---
def calculate_financial_metrics(data: UniversalFinancials) -> dict:
    metrics = data.model_dump()
    metrics["net_profit_margin_%"] = (data.net_income / data.total_revenue * 100) if data.total_revenue else 0
    metrics["return_on_equity_%"] = (data.net_income / data.total_equity * 100) if data.total_equity else 0
    metrics["debt_to_equity_ratio"] = (data.total_debt / data.total_equity) if data.total_equity else 0
    return metrics

async def run_pipeline(uploaded_files, llama_key, gemini_key):
    results = []
    for f in uploaded_files: # Sequential loop prevents 429 Resource Exhausted errors
        raw_md = await asyncio.to_thread(parse_pdf_cached, f.getvalue(), f.name, llama_key)
        json_str = normalize_with_llm_cached(raw_md, gemini_key)
        data = UniversalFinancials.model_validate_json(json_str)
        results.append(calculate_financial_metrics(data))
    return results

# --- 5. UI Logic ---
st.set_page_config(page_title="Financial Analyzer", layout="wide")
st.title("📈 Universal Corporate Financial Analyzer")
uploaded_pdfs = st.file_uploader("Upload Annual Reports (PDF)", type="pdf", accept_multiple_files=True)

if st.button("Run Universal Analysis") and uploaded_pdfs:
    results = asyncio.run(run_pipeline(uploaded_pdfs, LLAMA_CLOUD_KEY, GEMINI_API_KEY))
    df = pd.DataFrame(results).sort_values(by="fiscal_year")
    
    st.success("Analysis Complete!")
    st.dataframe(df.style.format(precision=2))
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(df, x="fiscal_year", y=["total_revenue", "net_income"], barmode="group", title="Growth"), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(df, x="fiscal_year", y="net_profit_margin_%", markers=True, title="Net Margin Trend"), use_container_width=True)
