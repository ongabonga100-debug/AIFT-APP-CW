import streamlit as st
import asyncio
import pandas as pd
import tempfile
import os
import requests
import time
import google.generativeai as genai
from pydantic import BaseModel

# DO NOT hardcode keys in production. Use st.secrets or environment variables.
LLAMA_CLOUD_KEY = st.secrets["LLAMA_CLOUD_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

class CanonicalFinancials(BaseModel):
    # These must be indented to belong to the class
    company_name: str
    fiscal_year: int
    revenue: float
    net_income: float
    total_assets: float
    total_equity: float
    current_assets: float
    current_liabilities: float
    total_debt: float

def parse_pdf_sync(file_path: str, llama_api_key: str) -> str:
    # Function bodies must be indented
    headers = {"accept": "application/json", "Authorization": f"Bearer {llama_api_key}"}
    with open(file_path, "rb") as f:
        # Code inside a 'with' block must be indented
        files = {"file": (os.path.basename(file_path), f, "application/pdf")}
        # Added the missing LlamaParse upload endpoint URL here
        upload_res = requests.post("https://api.cloud.llamaindex.ai/api/parsing/upload", headers=headers, files=files)
        upload_res.raise_for_status()
        job_id = upload_res.json()["id"]
        
        # NOTE: LlamaParse requires polling the job_id to get the final markdown. 
        # You will need to add a polling loop here to fetch the actual text.
        return f"Job ID: {job_id} (Requires polling implementation)"

async def parse_pdf_llamaparse(file_path: str, llama_api_key: str) -> str:
    return await asyncio.to_thread(parse_pdf_sync, file_path, llama_api_key)

def normalize_with_llm(raw_markdown: str, gemini_api_key: str) -> CanonicalFinancials:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    You are an expert financial controller. Extract the financial tables, resolve idiosyncratic naming, clean up formatting, and detect the scale.
    You MUST return ONLY a valid JSON object with exactly these keys and data types. Do not add any other text.
    {{
    "company_name": "string",
    "fiscal_year": 2023,
    "revenue": 0.0,
    "net_income": 0.0,
    "total_assets": 0.0,
    "total_equity": 0.0,
    "current_assets": 0.0,
    "current_liabilities": 0.0,
    "total_debt": 0.0
    }}
    Raw Report Markdown:
    {raw_markdown[:30000]}
    """
    response = model.generate_content(prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0.0))
    return CanonicalFinancials.model_validate_json(response.text)

def calculate_financial_metrics(data: CanonicalFinancials) -> dict:
    metrics = data.model_dump()
    net_profit_margin = data.net_income / data.revenue if data.revenue else 0
    asset_turnover = data.revenue / data.total_assets if data.total_assets else 0
    equity_multiplier = data.total_assets / data.total_equity if data.total_equity else 0
    roe = net_profit_margin * asset_turnover * equity_multiplier
    current_ratio = data.current_assets / data.current_liabilities if data.current_liabilities else 0
    debt_to_equity = data.total_debt / data.total_equity if data.total_equity else 0
    
    # Save the calculated metrics back to the dictionary
    metrics["net_profit_margin"] = net_profit_margin
    metrics["asset_turnover"] = asset_turnover
    metrics["equity_multiplier"] = equity_multiplier
    metrics["roe"] = roe
    metrics["current_ratio"] = current_ratio
    metrics["debt_to_equity"] = debt_to_equity
    
    return metrics

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

# Streamlit UI Setup
st.set_page_config(page_title="Multi-Entity Financial Analyzer", layout="wide")
st.title("AI-Driven Multi-Entity Financial Statement Analysis")
st.markdown("Upload annual reports (PDFs) to automatically parse, normalize, and benchmark financial health.")

uploaded_pdfs = st.file_uploader("Upload Company Annual Reports", type="pdf", accept_multiple_files=True)

if st.button("Run Concurrent Analysis"):
    # Nested blocks require further indentation
    if not uploaded_pdfs:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Extracting and analyzing reports..."):
            results = asyncio.run(main_pipeline(uploaded_pdfs, LLAMA_CLOUD_KEY, GEMINI_API_KEY))
            df = pd.DataFrame(results)
            df.set_index("company_name", inplace=True)
            st.dataframe(df) # Render the dataframe in the UI
