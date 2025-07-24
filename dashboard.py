import streamlit as st
import requests
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")

# Konstanta API
API_KEY = SECTORS_API_KEY
BASE_URL = "https://api.sectors.app/v1"
HEADERS = {"Authorization": API_KEY}

# Inisialisasi LLM
llm = ChatGroq(
    temperature=0.7,
    model_name="llama3-70b-8192",
    groq_api_key=GROQ_API_KEY
)

# --- SIDEBAR ---
st.sidebar.title("📌 Pilihan Analisis")
# Ambil subsektor
url_subsectors = f"{BASE_URL}/subsectors/"
resp = requests.get(url_subsectors, headers=HEADERS)
subsectors_df = pd.DataFrame(resp.json())
subsector_list = subsectors_df["subsector"].sort_values().tolist()
selected_subsector = st.sidebar.selectbox("🔽 Pilih Subsector", subsector_list)

# Ambil perusahaan berdasarkan subsektor
url_companies = f"{BASE_URL}/companies/"
params = {"sub_sector": selected_subsector}
resp = requests.get(url_companies, headers=HEADERS, params=params)
companies_df = pd.DataFrame(resp.json())
company_options = companies_df["symbol"] + " - " + companies_df["company_name"]
selected_company = st.sidebar.selectbox("🏢 Pilih Perusahaan", company_options)
symbol = selected_company.split(" - ")[0]

# Tombol lihat
if st.sidebar.button("🔍 Lihat Insight"):

    # --- 1. RINGKASAN PERUSAHAAN ---
    url_report = f"{BASE_URL}/company/report/{symbol}/?sections=overview"
    response_report = requests.get(url_report, headers=HEADERS)
    report_data = response_report.json()
    overview = report_data["overview"]

    with st.sidebar:
        st.markdown(f"### 💼 Ringkasan Perusahaan: {report_data['company_name']} ({symbol})")
        st.markdown(f"""
        - **Sektor/Subsektor**: {overview['sector']} / {overview['sub_sector']}
        - **Industri**: {overview['industry']}
        - **Board**: {overview['listing_board']}
        - **Tanggal Listing**: {overview['listing_date']}
        - **Jumlah Karyawan**: {int(overview.get('employee_num', 0)):,} orang
        - **Kapitalisasi Pasar**: Rp {overview['market_cap']:,.0f}
        - **Peringkat Kapitalisasi Pasar**: #{overview['market_cap_rank']}
        - **Harga Terakhir ({overview['latest_close_date']})**: Rp {overview['last_close_price']:,}
        - **Perubahan Harian**: {overview['daily_close_change']:.2%}
        - **All-Time High**: Rp {overview['all_time_price']['all_time_high']['price']:,} (pada {overview['all_time_price']['all_time_high']['date']})
        - **All-Time Low**: Rp {overview['all_time_price']['all_time_low']['price']:,} (pada {overview['all_time_price']['all_time_low']['date']})
        - **Website**: {overview['website']}
        - **Email IR**: [Link](mailto:{overview['email']})
        """)


    # --- 2. FINANCIAL SUMMARY (LLM) ---
    url_fin = f"{BASE_URL}/financials/quarterly/{symbol}/"
    params = {
        "report_date": "2025-06-30",
        "approx": "true",
        "n_quarters": "8"
    }
    response = requests.get(url_fin, headers=HEADERS, params=params)
    financials = pd.DataFrame(response.json())

    prompt_template = PromptTemplate.from_template("""
    Berikut adalah laporan keuangan kuartalan sebuah perusahaan (dalam miliar rupiah):

    {data}

    Tuliskan ringkasan keuangan dalam 3 kalimat, seolah-olah Anda menjelaskan kepada investor.
    """)
    prompt = prompt_template.format(data=financials)
    ringkasan_llm = llm.invoke(prompt)
    

    st.subheader("🤖 Ringkasan Keuangan oleh LLM")
    st.markdown(ringkasan_llm.content)

    # --- 3. TREN INTERPRETER (LLM) ---
    prompt_tren = f"""
    Berikut adalah data tren keuangan kuartalan perusahaan (dalam miliar rupiah):

    {financials}

    Analisis tren yang muncul dari data tersebut. Fokus pada perubahan revenue, net income, dan operating cashflow.
    Buat ringkasan dalam betuk 2-3 poin.
    """
    interpretasi_tren = llm.invoke(prompt_tren)
    st.subheader("🔎 Interpretasi Tren oleh LLM")
    st.markdown(interpretasi_tren.content)

    # --- 4. FINANCIAL RISK DETECTOR (LLM) ---
    prompt_risiko = f"""
    Berikut adalah data keuangan perusahaan:

    {financials.to_csv(index=False)}

    Tugas Anda adalah:
    - Identifikasi dan jelaskan risiko keuangan perusahaan ini.
    - Tulis dengan bahasa yang mudah dimengerti, seperti menjelaskan ke investor awam.
    - Sorot setiap risiko utama dengan simbol ❗
    - Tulis maksimal 3 poin.
    """
    risiko_llm = llm.invoke(prompt_risiko)
    st.subheader("⚠️ Deteksi Risiko Keuangan (LLM)")
    st.markdown(risiko_llm.content)
