import streamlit as st
import asyncio
import nest_asyncio
import pandas as pd
import tempfile
import os
import google.generativeai as genai
from pydantic import BaseModel, Field
from llama_parse import LlamaParse

# Apply nest_asyncio to allow asyncio execution within Streamlit's event loop
nest_asyncio.apply()

# -------------------------------------------------------------------
# API CONFIGURATION
# -------------------------------------------------------------------
LLAMA_CLOUD_KEY = "llx-c5LySHuQRqvfnZwiRMoBTdemxaT8VICBT2l6nPXzwny3Io4B"
GEMINI_API_KEY = "AIzaSyBwLn05CKZoN4JIf89MqSqsYr7bFcDu8y0"

# -------------------------------------------------------------------
# 1. DEFINE THE CANONICAL SCHEMA (PYDANTIC)
# -------------------------------------------------------------------
class CanonicalFinancials:
    company_name: str = Field(description="The name of the corporate entity.")
    fiscal_year: int = Field(description="The fiscal year of the report.")
    revenue: float = Field(description="Total top-line revenue or net sales.")
    net_income: float = Field(description="Bottom-line net income or net profit.")
    total_assets: float = Field(description="Total assets on the balance sheet.")
    total_equity: float = Field(description="Total stockholders' equity.")
    current_assets: float = Field(description="Total current assets.")
    current_liabilities: float = Field(description="Total current liabilities.")
    total_debt: float = Field(description="Short-term and long-term debt combined.")

# -------------------------------------------------------------------
# 2. CORE FUNCTIONS
# -------------------------------------------------------------------
async def parse_pdf_llamaparse(file_path: str, llama_api_key: str) -> str:
    """Extracts markdown (including tables) from complex PDFs using LlamaParse."""
    parser = LlamaParse(
        api_key=llama_api_key,
        result_type="markdown", 
        verbose=True
    )
    # Process asynchronously to handle multiple companies at once
    documents = await parser.aload_data(file_path)
    full_text = "\n".join([doc.text for doc in documents])
    return full_text

def normalize_with_llm(raw_markdown: str, gemini_api_key: str) -> CanonicalFinancials:
    """Uses Gemini and Pydantic to map idiosyncratic line items to a strict schema."""
    genai.configure(api_key=gemini_api_key)
    
    # We use Gemini 1.5 Flash for fast, structured extraction
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an expert financial controller. I am giving you the raw markdown extraction of a company's annual report.
    Your job is to read the financial tables, resolve idiosyncratic naming (e.g., mapping 'Net Sales' to 'Revenue'), 
    clean up formatting, detect the scale (e.g., multiply by 1,000 if 'in thousands'), and output the canonical metrics.
    
    Raw Report Markdown:
    {raw_markdown[:30000]} # Truncated for token limits in this prototype
    """
    
    # Enforce deterministic JSON output using our Pydantic schema
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=CanonicalFinancials,
            temperature=0.0
        )
    )
    
    # Parse the resulting JSON directly into our Pydantic object
    return CanonicalFinancials.model_validate_json(response.text)

def calculate_financial_metrics(data: CanonicalFinancials) -> dict:
    """Calculates DuPont Analysis and foundational ratios."""
    metrics = data.model_dump()
    
    # Profitability & Efficiency
    net_profit_margin = data.net_income / data.revenue if data.revenue else 0
    asset_turnover = data.revenue / data.total_assets if data.total_assets else 0
    equity_multiplier = data.total_assets / data.total_equity if data.total_equity else 0
    
    # DuPont ROE
    roe = net_profit_margin * asset_turnover * equity_multiplier
    
    # Liquidity & Solvency
    current_ratio = data.current_assets / data.current_liabilities if data.current_liabilities else 0
    debt_to_equity = data.total_debt / data.total_equity if data.total_equity else 0
    
    metrics.update({
        "Net Profit Margin (%)": round(net_profit_margin * 100, 2),
        "Asset Turnover (x)": round(asset_turnover, 2),
        "Equity Multiplier (x)": round(equity_multiplier, 2),
        "DuPont ROE (%)": round(roe * 100, 2),
        "Current Ratio (x)": round(current_ratio, 2),
        "Debt to Equity (x)": round(debt_to_equity, 2)
    })
    return metrics

async def process_company_report(uploaded_file, llama_key, gemini_key):
    """Orchestrates the pipeline for a single uploaded file."""
    # Save the uploaded Streamlit file to a temporary file for LlamaParse
    with tempfile.NamedNamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Step 1: Agentic OCR Extraction
        raw_md = await parse_pdf_llamaparse(tmp_path, llama_key)
        
        # Step 2: Canonical Normalization
        canonical_data = normalize_with_llm(raw_md, gemini_key)
        
        # Step 3: Computational Analysis
        final_metrics = calculate_financial_metrics(canonical_data)
        return final_metrics
    finally:
        os.unlink(tmp_path)

async def main_pipeline(uploaded_files, llama_key, gemini_key):
    """Processes all uploaded reports concurrently."""
    tasks = [process_company_report(f, llama_key, gemini_key) for f in uploaded_files]
    return await asyncio.gather(*tasks)

# -------------------------------------------------------------------
# 3. STREAMLIT USER INTERFACE
# -------------------------------------------------------------------
st.set_page_config(page_title="Multi-Entity Financial Analyzer", layout="wide")
st.title("AI-Driven Multi-Entity Financial Statement Analysis")
st.markdown("Upload annual reports (PDFs) to automatically parse, normalize, and benchmark financial health.")

# Multi-file uploader for concurrent analysis
uploaded_pdfs = st.file_uploader(
    "Upload Company Annual Reports (PDFs)", 
    type="pdf", 
    accept_multiple_files=True 
)
