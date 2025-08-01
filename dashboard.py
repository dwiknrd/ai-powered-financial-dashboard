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
BASE_URL = "https://api.sectors.app/v1"
HEADERS = {"Authorization": SECTORS_API_KEY}

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

# Opsi perusahaan
company_options = companies_df["symbol"] + " - " + companies_df["company_name"]
selected_company = st.sidebar.selectbox("🏢 Pilih Perusahaan", company_options)

# Symbol perusahaan
symbol = selected_company.split(" - ")[0]



# Tombol lihat
if st.sidebar.button("🔍 Lihat Insight"):


    # --- 1. RINGKASAN PERUSAHAAN ---
    url_report = f"{BASE_URL}/company/report/{symbol}/?sections=overview"
    response_report = requests.get(url_report, headers=HEADERS)
    report_data = response_report.json()
    overview = report_data.get("overview", {})
    company_name = report_data.get("company_name", "N/A")

    with st.sidebar:
        st.markdown("---")
        st.markdown(f"### 💼 Ringkasan Perusahaan: {company_name} ({symbol})")
        st.markdown(f"""
        - **Sektor/Industri**: {overview.get('sector', 'N/A')} / {overview.get('industry', 'N/A')}
        - **Tanggal Listing**: {overview.get('listing_date', 'N/A')}
        - **Jumlah Karyawan**: {int(overview.get('employee_num', 0)):,} orang

        ---
        """)
        st.markdown("### 📊 Kinerja Pasar")
        st.markdown(f"""
        - **Kapitalisasi Pasar**: Rp {overview.get('market_cap', 0):,.0f} (Peringkat #{overview.get('market_cap_rank', 'N/A')})
        - **Harga Terakhir ({overview.get('latest_close_date', 'N/A')})**: Rp {overview.get('last_close_price', 0):,}
        - **Perubahan Harian**: {overview.get('daily_close_change', 0):.2%}
        - **Website**: {overview.get('website', 'N/A')}
        ---
        """)

    st.markdown(f"## 📈 Analisis Keuangan Perusahaan: {company_name} ({symbol})")

    ####### ---------- OPTIONAL SECTIONS ---------- ########
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Kapitalisasi Pasar", f"Rp {overview.get('market_cap', 0):,.0f}",
                      delta=f"{overview.get('daily_close_change', 0):.2%}",
                      delta_color="inverse")
            st.metric("Harga Terakhir", f"Rp {overview.get('last_close_price', 0):,.2f}",
                      delta=f"{overview.get('daily_close_change', 0):.2%}",
                      delta_color="inverse")
        with col2:
            st.metric("Jumlah Karyawan", f"{int(overview.get('employee_num', 0)):,} orang",
                      delta=f"{overview.get('employee_num_change', 0):.2%}",
                      delta_color="inverse")
            st.metric("Tanggal Listing", overview.get('listing_date', 'N/A'),
                      delta_color="off" )
    #### --- 2. FINANCIAL SUMMARY (LLM) ---
    url_fin = f"{BASE_URL}/financials/quarterly/{symbol}/"
    params = {"n_quarters": "4",
            "report_date": "2023-09-30"}

    response = requests.get(url_fin, headers=HEADERS, params=params)
    financials = pd.DataFrame(response.json())

    prompt_summary_template = PromptTemplate.from_template(
    """
    Anda adalah seorang analis keuangan yang handal.
    Berdasarkan data keuangan kuartalan berikut (dalam miliar Rupiah):
    ---
    {data}
    ---
    Tuliskan ringkasan eksekutif dalam 3 poin singkat untuk seorang investor. 
    Fokus pada: 
    1. Tren pertumbuhan pendapatan (revenue).
    2. Tingkat profitabilitas (net income).
    3. Posisi arus kas operasi (operating cashflow).
    """
    )
    prompt_summary = prompt_summary_template.format(data=financials.to_string(index=False))
    summary_llm = llm.invoke(prompt_summary)
    
    with st.expander("💡 Ringkasan Keuangan"):
        st.markdown(summary_llm.content)

    ### --- 3. REVENUE TRENDS ---
    sample_data_viz = financials[['date', 'revenue']].dropna().to_string(index=False)
    template_viz = PromptTemplate.from_template(
    """
    Anda adalah seorang programmer Python yang ahli dalam visualisasi data.

    Berikut adalah data pendapatan (revenue) perusahaan:
    ---
    {data}
    ---

    Buat sebuah skrip Python menggunakan **matplotlib** untuk menghasilkan **line plot**.

    Instruksi:
    - Sumbu X adalah 'date'.
    - Sumbu Y adalah 'revenue'.
    - Judul grafik: 'Tren Pendapatan Kuartalan untuk {symbol}'.
    - Ukuran gambar: (10, 6).
    - Format tanggal di sumbu X agar tidak tumpang tindih (gunakan `autofmt_xdate()`).
    - Simpan figure sebagai variabel bernama `fig`.
    - Jangan gunakan `plt.show()`.

    Tulis HANYA kode Python yang bisa langsung dieksekusi. Jangan sertakan penjelasan atau output apapun.
    """
    )

    llm_viz = llm.invoke(template_viz.format(data=sample_data_viz, symbol=symbol))
    clean_code = llm_viz.content.strip().strip("```").replace("python", "").strip()
    with st.expander("📊 Visualisasi Tren Pendapatan"):
        st.code(clean_code, language="python")
        st.pyplot(exec(clean_code))

    #### --- 4. FINANCIAL TRENDS ---
    template_trend = PromptTemplate.from_template(
    """
    Bertindaklah sebagai seorang analis keuangan. 
    Perhatikan data keuangan kuartalan berikut:
    ---
    {data}
    ---
    Analisis tren utama yang muncul dari data tersebut. Fokus pada pergerakan **revenue, net_income, dan operating_cashflow** dari kuartal ke kuartal.
    Sajikan analisis Anda dalam 3 poin (bullet points). Tuliskan dengan bahasa singkat dan jelas.
    """
    )
    interpretasi_tren = llm.invoke(template_trend.format(data=financials.to_string(index=False)))
    with st.expander("🔎 Interpretasi Tren Keuangan"):
        st.markdown(interpretasi_tren.content)

    #### --- 5. POTENTIAL FINANCIAL RISKS ---
    template_risk = PromptTemplate.from_template(
    """
    Anda adalah seorang analis risiko keuangan yang skeptis. 
    Periksa data keuangan berikut dengan teliti:
    ---
    {data}
    ---
    Identifikasi 2-3 potensi risiko atau "red flags" yang perlu diwaspadai dari data ini. Carilah hal-hal seperti penurunan margin, peningkatan utang, atau arus kas yang tidak sejalan dengan laba.
    Jelaskan setiap risiko dalam satu kalimat singkat.
    """
    )  
    llm_risiko = llm.invoke(template_risk.format(data=financials.to_string(index=False)))
    with st.expander("⚠️ Potensi Risiko Keuangan"):
        st.markdown(llm_risiko.content)