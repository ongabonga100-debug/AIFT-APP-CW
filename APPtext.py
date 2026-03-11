import streamlit as st
import asyncio
import pandas as pd
import tempfile
import os
import requests
import time
import json
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel

# --- SECURE KEY RETRIEVAL ---
LLAMA_CLOUD_KEY = st.secrets["LLAMA_CLOUD_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# --- 1. UNIVERSAL DATA MODEL ---
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

# --- 2. 2025-2026 SECTOR BENCHMARKS ---
BENCHMARKS = {
    "Banking": {"ROE_%": 14.1, "Net_Margin_%": 30.8, "Asset_Turnover": 0.1, "Debt_Equity": 8.5},
    "Technology": {"ROE_%": 18.5, "Net_Margin_%": 19.1, "Asset_Turnover": 0.8, "Debt_Equity": 0.2},
    "Manufacturing": {"ROE_%": 12.5, "Net_Margin_%": 8.2, "Asset_Turnover": 1.2, "Debt_Equity": 0.6},
    "FMCG": {"ROE_%": 25.0, "Net_Margin_%": 15.0, "Asset_Turnover": 1.5, "Debt_Equity": 0.1},
    "Retail": {"ROE_%": 16.0, "Net_Margin_%": 4.5, "Asset_Turnover": 2.2, "Debt_Equity": 0.8}
}

# --- 3. PROCESSING PIPELINE ---
async def process_report(uploaded_file, llama_key, gemini_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    try:
        # LlamaParse Extraction
        headers = {"accept": "application/json", "Authorization": f"Bearer {llama_key}"}
        upload = requests.post("https://api.cloud.llamaindex.ai/api/parsing/upload", 
                               headers=headers, files={"file": open(path, "rb")})
        job_id = upload.json()["id"]
        while True:
            time.sleep(3)
            status = requests.get(f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}", headers=headers).json()["status"]
            if status == "SUCCESS": break
            if status == "ERROR": raise Exception("Parsing Failed")
        
        raw_md = requests.get(f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/markdown", headers=headers).json()["markdown"]
        
        # Gemini Extraction with strict JSON guidance
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        Extract financial metrics from the following markdown. 
        Return ONLY a FLAT JSON object. Do not nest it inside keys like 'financials'.
        If a value is missing, use 0.0.
        
        Fields required: company_name (str), fiscal_year (int), industry_category (str), total_revenue (float), 
        operating_expenses (float), net_income (float), total_assets (float), total_liabilities (float), 
        total_equity (float), total_debt (float), operating_cash_flow (float).
        
        Markdown text:
        {raw_md[:60000]}
        """
        response = model.generate_content(prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json"))
        
        # Robust validation: Handle nested JSON if Gemini ignores instructions
        raw_json = json.loads(response.text)
        if isinstance(raw_json, dict) and len(raw_json) == 1:
            first_val = next(iter(raw_json.values()))
            if isinstance(first_val, dict):
                raw_json = first_val
                
        data = UniversalFinancials.model_validate(raw_json)
        
        # Ratio Engineering
        m = data.model_dump()
        m["net_margin_%"] = (data.net_income / data.total_revenue * 100) if data.total_revenue else 0
        m["roe_%"] = (data.net_income / data.total_equity * 100) if data.total_equity else 0
        m["asset_turnover"] = data.total_revenue / data.total_assets if data.total_assets else 0
        m["equity_multiplier"] = data.total_assets / data.total_equity if data.total_equity else 1
        m["debt_to_equity"] = data.total_debt / data.total_equity if data.total_equity else 0
        return m
    finally:
        os.unlink(path)

# --- 4. AI ANALYST ---
def generate_ai_analysis(df: pd.DataFrame, gemini_key: str) -> str:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    industry = df['industry_category'].iloc[-1]
    bench = BENCHMARKS.get(industry, BENCHMARKS["Manufacturing"])
    
    prompt = f"Perform a comparative analysis for these entities. Industry Norms (2026): ROE {bench['ROE_%']}%, Margin {bench['Net_Margin_%']}%. Identify leaders. Data: {df.to_markdown()}"
    return model.generate_content(prompt).text

# --- 5. STREAMLIT DASHBOARD ---
st.set_page_config(page_title="Max Financial Analyzer", layout="wide")
st.title("🏦 Max-Analysis Intelligence Suite")

files = st.file_uploader("Upload Annual Reports (PDF)", type="pdf", accept_multiple_files=True)

if st.button("Execute Strategic Analysis"):
    if not files:
        st.warning("Please upload reports.")
    else:
        with st.spinner("Analyzing reports..."):
            # SAFE ASYNCIO PATTERN FOR STREAMLIT
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            results = loop.run_until_complete(asyncio.gather(*[process_report(f, LLAMA_CLOUD_KEY, GEMINI_API_KEY) for f in files]))
            df = pd.DataFrame(results).sort_values(["company_name", "fiscal_year"])
            
            # --- DASHBOARD ---
            st.success("Analysis Complete!")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", data=csv, file_name="analysis.csv", use_container_width=True)
            
            t1, t2, t3 = st.tabs(["📊 Performance", "🧬 DuPont Analysis", "🤖 AI Scorecard"])
            
            with t1:
                st.plotly_chart(px.line(df, x="fiscal_year", y="total_revenue", color="company_name", markers=True, title="Revenue Trends"), use_container_width=True)
                st.plotly_chart(px.bar(df, x="fiscal_year", y="net_margin_%", color="company_name", barmode="group", title="Profitability Benchmarks"), use_container_width=True)
            
            with t2:
                st.subheader("DuPont ROE DNA")
                
                fig = px.scatter(df, x="asset_turnover", y="net_margin_%", size="equity_multiplier", color="company_name", hover_name="fiscal_year", title="Efficiency vs Profitability (Bubble = Leverage)")
                st.plotly_chart(fig, use_container_width=True)
            
            with t3:
                st.subheader("2026 Sector Scorecard")
                st.markdown(generate_ai_analysis(df, GEMINI_API_KEY))
