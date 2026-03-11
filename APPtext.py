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

# --- 2. LlamaParse Extraction ---
def parse_pdf_sync(file_path: str, llama_api_key: str) -> str:
    headers = {"accept": "application/json", "Authorization": f"Bearer {llama_api_key}"}
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/pdf")}
        upload_res = requests.post("https://api.cloud.llamaindex.ai/api/parsing/upload", headers=headers, files=files)
        upload_res.raise_for_status()
        job_id = upload_res.json()["id"]
        
        while True:
            time.sleep(3)
            status_res = requests.get(f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}", headers=headers)
            status_res.raise_for_status()
            status = status_res.json()["status"]
            
            if status == "SUCCESS":
                result_res = requests.get(f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/markdown", headers=headers)
                result_res.raise_for_status()
                return result_res.json()["markdown"]
            elif status == "ERROR":
                raise Exception(f"LlamaParse failed for job {job_id}")

async def parse_pdf_llamaparse(file_path: str, llama_api_key: str) -> str:
    return await asyncio.to_thread(parse_pdf_sync, file_path, llama_api_key)

# --- 3. LLM Normalization ---
def normalize_with_llm(raw_markdown: str, gemini_api_key: str) -> UniversalFinancials:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = f"""
    You are an expert corporate financial analyst. Scan the provided Markdown text of this company's annual report.
    Because companies operate in different industries, you must intelligently map their specific accounting terms to universal metrics.
    
    Rules for Mapping:
    - If it is a Bank: 'total_revenue' = Interest Earned + Other Income. 'operating_expenses' = Operating Expenses + Provisions. 'total_debt' = Borrowings.
    - If it is a Standard Corporation: 'total_revenue' = Total Sales/Revenue.
    - If a specific metric truly cannot be found or calculated from the text, use 0.0.
    
    Extract the metrics and return ONLY a valid JSON object matching exactly this structure:
    {{
    "company_name": "Actual Company Name",
    "fiscal_year": 2024,
    "industry_category": "e.g., Banking, Technology, Manufacturing, Retail",
    "total_revenue": 0.0,
    "operating_expenses": 0.0,
    "net_income": 0.0,
    "total_assets": 0.0,
    "total_liabilities": 0.0,
    "total_equity": 0.0,
    "total_debt": 0.0,
    "operating_cash_flow": 0.0
    }}
    
    Raw Report Markdown:
    {raw_markdown}
    """
    
    response = model.generate_content(
        prompt, 
        generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0.1)
    )
    return UniversalFinancials.model_validate_json(response.text)

# --- 4. Universal KPI Calculations ---
def calculate_financial_metrics(data: UniversalFinancials) -> dict:
    metrics = data.model_dump()
    
    metrics["net_profit_margin_%"] = (data.net_income / data.total_revenue * 100) if data.total_revenue else 0
    metrics["operating_margin_%"] = ((data.total_revenue - data.operating_expenses) / data.total_revenue * 100) if data.total_revenue else 0
    metrics["return_on_assets_%"] = (data.net_income / data.total_assets * 100) if data.total_assets else 0
    metrics["return_on_equity_%"] = (data.net_income / data.total_equity * 100) if data.total_equity else 0
    metrics["asset_turnover"] = (data.total_revenue / data.total_assets) if data.total_assets else 0
    metrics["debt_to_equity_ratio"] = (data.total_debt / data.total_equity) if data.total_equity else 0
    metrics["liabilities_to_assets_%"] = (data.total_liabilities / data.total_assets * 100) if data.total_assets else 0
    
    return metrics

# --- 5. AI Executive Summary Generator ---
def generate_executive_summary(df: pd.DataFrame, gemini_api_key: str) -> str:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Convert the DataFrame to a Markdown table string so the LLM can read the numbers
    data_string = df.to_markdown(index=False)
    
    prompt = f"""
    You are a Chief Financial Officer and expert corporate strategist. 
    Review the following multi-year financial data for {df['company_name'].iloc[-1]}.
    
    Write a concise, highly analytical executive summary suitable for a high-level MBA case study or capstone presentation. 
    Structure your response into these 3 sections:
    1. **Top-Line & Bottom-Line Performance:** Discuss revenue growth and net income trends over the years provided.
    2. **Operational Efficiency:** Analyze profitability margins, Return on Assets (ROA), and Return on Equity (ROE).
    3. **Solvency & Risk:** Evaluate the debt-to-equity ratio and overall financial leverage.
    
    Keep it professional, objective, and STRICTLY based on the provided numbers. 
    
    Extracted Financial Data:
    {data_string}
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- 6. Async Pipeline Execution ---
async def process_company_report(uploaded_file, llama_key, gemini_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        raw_md = await parse_pdf_llamaparse(tmp_path, llama_key)
        canonical_data = normalize_with_llm(raw_md, gemini_key)
        return calculate_financial_metrics(canonical_data)
    finally:
        os.unlink(tmp_path)

async def main_pipeline(uploaded_files, llama_key, gemini_key):
    tasks = [process_company_report(f, llama_key, gemini_key) for f in uploaded_files]
    return await asyncio.gather(*tasks)

# --- 7. Streamlit UI & Dashboard ---
st.set_page_config(page_title="Universal Financial Analyzer", layout="wide")
st.title("📈 Universal Corporate Financial Analyzer")
st.markdown("Upload annual reports to extract KPIs, visualize financial health, and generate AI insights.")

uploaded_pdfs = st.file_uploader("Upload Annual Reports (PDFs)", type="pdf", accept_multiple_files=True)

if st.button("Run Universal Analysis"):
    if not uploaded_pdfs:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Extracting and mapping financial data (this may take a minute)..."):
            results = asyncio.run(main_pipeline(uploaded_pdfs, LLAMA_CLOUD_KEY, GEMINI_API_KEY))
            df = pd.DataFrame(results)
            
            if "fiscal_year" in df.columns:
                df = df.sort_values(by="fiscal_year")
            
            st.success("Analysis Complete!")
            
            # Overview & Download Section
            st.subheader(f"Overview: {df['company_name'].iloc[-1]} ({df['industry_category'].iloc[-1]})")
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            
            col_data, col_download = st.columns([3, 1])
            with col_download:
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv_data,
                    file_name=f"{df['company_name'].iloc[-1]}_financials.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with st.expander("View Raw Extracted Data", expanded=True):
                st.dataframe(df.style.format(precision=2))
            
            st.divider()
            
            # Dashboard
            st.subheader("📊 Financial Health Dashboard")
            
            col1, col2 = st.columns(2)
            with col1:
                fig_income = px.bar(
                    df, x="fiscal_year", y=["total_revenue", "net_income"], barmode="group",
                    color_discrete_sequence=["#1f77b4", "#2ca02c"], title="Top & Bottom Line Growth"
                )
                st.plotly_chart(fig_income, use_container_width=True)

            with col2:
                fig_margin = px.line(
                    df, x="fiscal_year", y=["net_profit_margin_%", "operating_margin_%"], markers=True,
                    color_discrete_sequence=["#ff7f0e", "#d62728"], title="Profitability Margins"
                )
                st.plotly_chart(fig_margin, use_container_width=True)
                
            col3, col4 = st.columns(2)
            with col3:
                fig_returns = px.bar(
                    df, x="fiscal_year", y=["return_on_assets_%", "return_on_equity_%"], barmode="group",
                    color_discrete_sequence=["#9467bd", "#8c564b"], title="Returns Profile (ROA & ROE)"
                )
                st.plotly_chart(fig_returns, use_container_width=True)

            with col4:
                fig_leverage = px.area(
                    df, x="fiscal_year", y="debt_to_equity_ratio", color_discrete_sequence=["#7f7f7f"],
                    title="Leverage (Debt-to-Equity Ratio)"
                )
                st.plotly_chart(fig_leverage, use_container_width=True)
            
            st.divider()
            
            # AI Executive Summary
            st.subheader("🤖 AI Executive Summary")
            with st.spinner("Drafting strategic analysis..."):
                summary = generate_executive_summary(df, GEMINI_API_KEY)
                st.write(summary)
