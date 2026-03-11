import streamlit as st
import asyncio
import pandas as pd
import tempfile
import os
import requests
import time
import google.generativeai as genai
from pydantic import BaseModel

API CONFIGURATION
LLAMA_CLOUD_KEY = "llx-c5LySHuQRqvfnZwiRMoBTdemxaT8VICBT2l6nPXzwny3Io4B"
GEMINI_API_KEY = "AIzaSyBwLn05CKZoN4JIf89MqSqsYr7bFcDu8y0"

class CanonicalFinancials(BaseModel):
company_name: str
fiscal_year: int
revenue: float
net_income: float
total_assets: float
total_equity: float
current_assets: float
current_liabilities: float
total_debt: float

def _parse_pdf_sync(file_path: str, llama_api_key: str) -> str:
headers = {
"accept": "application/json",
"Authorization": f"Bearer {llama_api_key}"
}
with open(file_path, "rb") as f:
files = {"file": (os.path.basename(file_path), f, "application/pdf")}
upload_res = requests.post(
"",
headers=headers,
files=files
)
upload_res.raise_for_status()
job_id = upload_res.json()["id"]

async def parse_pdf_llamaparse(file_path: str, llama_api_key: str) -> str:
return await asyncio.to_thread(_parse_pdf_sync, file_path, llama_api_key)

def normalize_with_llm(raw_markdown: str, gemini_api_key: str) -> CanonicalFinancials:
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')
prompt = f"""
You are an expert financial controller. I am giving you the raw markdown extraction of a company's annual report.
Your job is to read the financial tables, resolve idiosyncratic naming, clean up formatting, and detect the scale.
You MUST return ONLY a valid JSON object with exactly these keys and data types. Do not add any other text.
{{
"company_name": "string",
"fiscal_year": "integer",
"revenue": "float",
"net_income": "float",
"total_assets": "float",
"total_equity": "float",
"current_assets": "float",
"current_liabilities": "float",
"total_debt": "float"
}}

def calculate_financial_metrics(data: CanonicalFinancials) -> dict:
metrics = data.model_dump()
net_profit_margin = data.net_income / data.revenue if data.revenue else 0
asset_turnover = data.revenue / data.total_assets if data.total_assets else 0
equity_multiplier = data.total_assets / data.total_equity if data.total_equity else 0
roe = net_profit_margin * asset_turnover * equity_multiplier
current_ratio = data.current_assets / data.current_liabilities if data.current_liabilities else 0
debt_to_equity = data.total_debt / data.total_equity if data.total_equity else 0

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

st.set_page_config(page_title="Multi-Entity Financial Analyzer", layout="wide")
st.title("AI-Driven Multi-Entity Financial Statement Analysis")
st.markdown("Upload annual reports (PDFs) to automatically parse, normalize, and benchmark financial health.")

uploaded_pdfs = st.file_uploader("Upload Company Annual Reports (PDFs)", type="pdf", accept_multiple_files=True)

if st.button("Run Concurrent Analysis"):
if not uploaded_pdfs:
st.warning("Please upload at least one annual report PDF.")
else:
with st.spinner(f"Extracting and analyzing {len(uploaded_pdfs)} reports concurrently..."):
results = asyncio.run(main_pipeline(uploaded_pdfs, LLAMA_CLOUD_KEY, GEMINI_API_KEY))
df = pd.DataFrame(results)
df.set_index("company_name", inplace=True)
