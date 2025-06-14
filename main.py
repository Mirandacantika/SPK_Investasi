import streamlit as st
import pandas as pd
import numpy as np

# === PAGE CONFIG ===
st.set_page_config(page_title="SPK Investasi Mahasiswa", layout="wide")

# === BAHASA ===
if 'language' not in st.session_state:
    st.session_state.language = 'id'
lang = st.session_state.language

labels = {
    'id': {
        'title': "\U0001F4CA Sistem Pendukung Keputusan Investasi Usaha Mahasiswa",
        'manual': "\U0001F4DD Input Manual",
        'upload': "\U0001F4C1 Upload File",
        'num_usaha': "Jumlah Usaha",
        'save': "\U0001F4BE Simpan & Tampilkan Hasil",
        'download_template': "\u2B07\ufe0f Unduh Template Kosong (CSV)",
        'upload_prompt': "Unggah file CSV",
        'data_usaha': "\U0001F4C4 Data Usaha Mahasiswa",
        'bobot': "\U0001F4CC Bobot Kriteria (Metode CRITIC)",
        'hasil': "\U0001F4C8 Hasil Rekomendasi Investasi",
        'download_hasil': "\U0001F4BE Unduh Hasil",
        'change_lang': "\U0001F1FA\U0001F1F8 English",
        'kriteria': ['ROI (%)', 'Modal Awal (Rp)', 'Pendapatan Rata-Rata 3 Bulan (Rp)', 'Aset (Rp)',
                     'Inovasi Produk (1-5)', 'Peluang Pasar (1-5)', 'Tingkat Risiko (1-5)'],
        'nama_usaha': 'Nama Usaha',
        'status': {
            'sangat_layak': "Sangat Layak",
            'layak': "Layak",
            'cukup_layak': "Cukup Layak",
            'kurang_layak': "Kurang Layak",
            'tidak_layak': "Tidak Layak"
        }
    },
    'en': {
        'title': "\U0001F4CA Decision Support System for Student Business Investment",
        'manual': "\U0001F4DD Manual Input",
        'upload': "\U0001F4C1 Upload File",
        'num_usaha': "Number of Businesses",
        'save': "\U0001F4BE Save & Show Result",
        'download_template': "\u2B07\ufe0f Download Blank Template (CSV)",
        'upload_prompt': "Upload CSV file",
        'data_usaha': "\U0001F4C4 Student Business Data",
        'bobot': "\U0001F4CC Criteria Weights (CRITIC Method)",
        'hasil': "\U0001F4C8 Investment Recommendation Result",
        'download_hasil': "\U0001F4BE Download Result",
        'change_lang': "\U0001F1EE\U0001F1E9 Bahasa Indonesia",
        'kriteria': ['ROI (%)', 'Initial Capital (Rp)', 'Avg. 3-Month Revenue (Rp)', 'Assets (Rp)',
                     'Product Innovation (1-5)', 'Market Opportunity (1-5)', 'Risk Level (1-5)'],
        'nama_usaha': 'Business Name',
        'status': {
            'sangat_layak': "Highly Recommended",
            'layak': "Recommended",
            'cukup_layak': "Moderately Recommended",
            'kurang_layak': "Less Recommended",
            'tidak_layak': "Not Recommended"
        }
    }
}

standard_kriteria = ['ROI (%)', 'Modal Awal (Rp)', 'Pendapatan Rata-Rata 3 Bulan (Rp)',
                     'Aset (Rp)', 'Inovasi Produk (1-5)', 'Peluang Pasar (1-5)', 'Tingkat Risiko (1-5)']

# === STYLING ===
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #EAF4FF;
        border-right: 1px solid #D0E3F1;
    }
    div.stButton > button {
        width: 100%;
        background-color: #2196F3;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        padding: 0.6em 1.2em;
        margin-bottom: 10px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #0b7dda;
    }
    .stDownloadButton button {
        width: 100%;
        background-color: #00BFFF;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        margin-bottom: 10px;
    }
    .dataframe th {
        background-color: #F0F8FF;
    }
    .dataframe td {
        text-align: center;
        padding: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.title("SPK Investasi Mahasiswa")
manual_click = st.sidebar.button(labels[lang]['manual'], key="btn_manual")
upload_click = st.sidebar.button(labels[lang]['upload'], key="btn_upload")
with st.sidebar:
    st.markdown("---")
    if st.button(labels[lang]['change_lang']):
        st.session_state.language = 'en' if lang == 'id' else 'id'
        st.rerun()

if 'input_method' not in st.session_state:
    st.session_state.input_method = "Manual"
if manual_click:
    st.session_state.input_method = "Manual"
if upload_click:
    st.session_state.input_method = "Upload"
input_method = st.session_state.input_method
