import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from scipy.spatial.distance import euclidean as euclid_dist, cityblock

# === PAGE CONFIG ===
st.set_page_config(page_title="SPK Investasi Mahasiswa", layout="wide")

# === BAHASA ===
if 'language' not in st.session_state:
    st.session_state.language = 'id'
lang = st.session_state.language

labels = {
    'id': {
        'title': "📊 Sistem Pendukung Keputusan Investasi Usaha Mahasiswa",
        'manual': "📝 Input Manual",
        'upload': "📁 Upload File",
        'num_usaha': "Jumlah Usaha",
        'save': "💾 Simpan & Tampilkan Hasil",
        'download_template': "⬇️ Unduh Template Kosong (CSV)",
        'upload_prompt': "Unggah file CSV",
        'data_usaha': "📄 Data Usaha Mahasiswa",
        'bobot': "📌 Bobot Kriteria (Metode CRITIC)",
        'hasil': "📈 Hasil Rekomendasi Investasi",
        'download_hasil': "💾 Unduh Hasil",
        'change_lang': "🇬🇧 English",
        'kriteria': [
            'ROI (%)',
            'Modal Awal (Rp)',
            'Pendapatan Rata-Rata 3 Bulan (Rp)',
            'Aset (Rp)',
            'Inovasi Produk (1-5)',
            'Peluang Pasar (1-5)',
            'Tingkat Risiko (1-5)'
        ],
        'nama_usaha': 'Nama Usaha',
        'status': {
            'sangat_layak': "Sangat Layak",
            'layak': "Layak",
            'cukup_layak': "Cukup Layak",
            'kurang_layak': "Kurang Layak",
            'tidak_layak': "Tidak Layak"
        },
        'error_kolom': "❗ Kolom pada file tidak sesuai dengan format yang diharapkan."
    }
}

# === CSS STYLING ===
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
manual_click = st.sidebar.button(labels[lang]['manual'])
upload_click = st.sidebar.button(labels[lang]['upload'])
if st.sidebar.button(labels[lang]['change_lang']):
    st.session_state.language = 'en' if lang == 'id' else 'id'
    st.rerun()

if 'input_method' not in st.session_state:
    st.session_state.input_method = "Manual"
if manual_click:
    st.session_state.input_method = "Manual"
if upload_click:
    st.session_state.input_method = "Upload"
input_method = st.session_state.input_method

# === FUNGSI PERHITUNGAN ===
def calculate_critic(data, cost_cols=[]):
    norm = data.copy()
    for col in data.columns:
        if col in cost_cols:
            norm[col] = data[col].min() / data[col]
        else:
            norm[col] = data[col] / data[col].max()
    std_dev = norm.std()
    corr_matrix = norm.corr()
    conflict = 1 - corr_matrix.abs()
    info = std_dev * conflict.sum()
    weights = info / info.sum()
    return weights, norm

def calculate_codas_full(norm_data, weights):
    # Weighted normalized matrix
    weighted = norm_data * weights
    # Negative ideal
    ideal_neg = weighted.min()
    # Distances
    eu_dist = weighted.apply(lambda row: euclid_dist(row, ideal_neg), axis=1)
    tb_dist = weighted.apply(lambda row: cityblock(row, ideal_neg), axis=1)
    # Ra scoring
    mu = 0.02
    scores = []
    for i in range(len(weighted)):
        ra_i = 0
        for j in range(len(weighted)):
            if i == j:
                continue
            e_diff = eu_dist[i] - eu_dist[j]
            t_diff = tb_dist[i] - tb_dist[j]
            hik = e_diff if abs(e_diff) > mu else e_diff + mu * t_diff
            ra_i += hik
        scores.append(ra_i)
    return pd.Series(scores, index=norm_data.index)

def get_status(score, modal):
    if score >= 0.81:
        return labels[lang]['status']['sangat_layak'], modal * 0.60
    elif score >= 0.61:
        return labels[lang]['status']['layak'], modal * 0.45
    elif score >= 0.41:
        return labels[lang]['status']['cukup_layak'], modal * 0.30
    elif score >= 0.21:
        return labels[lang]['status']['kurang_layak'], modal * 0.15
    else:
        return labels[lang]['status']['tidak_layak'], 0.0

def validate_csv_columns(df, expected_cols):
    return all(col in df.columns for col in expected_cols)

# === MAIN LOGIC ===
st.title(labels[lang]['title'])
df_usaha = None

# Input manual
if input_method == "Manual":
    st.subheader(labels[lang]['manual'])
    num = st.number_input(labels[lang]['num_usaha'], min_value=1, max_value=20, step=1)
    default_data = pd.DataFrame({
        labels[lang]['nama_usaha']: [f"Usaha {i+1}" for i in range(num)],
        **{col: [0.0]*num for col in labels[lang]['kriteria']}
    })
    df_input = st.data_editor(default_data, use_container_width=True, num_rows="dynamic")
    if st.button(labels[lang]['save']):
        df_usaha = df_input.copy()

# Input via CSV
elif input_method == "Upload":
    st.subheader(labels[lang]['upload'])
    template_df = pd.DataFrame({
        labels[lang]['nama_usaha']: [""],
        **{col: [0.0] for col in labels[lang]['kriteria']}
    })
    st.download_button(
        label=labels[lang]['download_template'],
        data=template_df.to_csv(index=False).encode('utf-8'),
        file_name='template_input_usaha.csv',
        mime='text/csv'
    )
    uploaded_file = st.file_uploader(labels[lang]['upload_prompt'], type=["csv"])
    if uploaded_file:
        df_candidate = pd.read_csv(uploaded_file)
        expected_cols = [labels[lang]['nama_usaha']] + labels[lang]['kriteria']
        if validate_csv_columns(df_candidate, expected_cols):
            df_usaha = df_candidate.copy()
        else:
            st.error(labels[lang]['error_kolom'])

# Proses & tampilkan
if df_usaha is not None:
    st.subheader(labels[lang]['data_usaha'])
    st.dataframe(df_usaha, use_container_width=True)

    # Hitung CRITIC
    df_kriteria = df_usaha[labels[lang]['kriteria']].apply(pd.to_numeric, errors='coerce').fillna(0)
    weights, df_norm = calculate_critic(
        df_kriteria,
        cost_cols=["Modal Awal (Rp)", "Tingkat Risiko (1-5)"]
    )
    st.subheader(labels[lang]['bobot'])
    st.write(weights)

    # Hitung CODAS
    df_usaha["Skor CODAS"] = calculate_codas_full(df_norm, weights)
    df_usaha["Peringkat"] = df_usaha["Skor CODAS"].rank(ascending=False, method='min').astype(int)
    df_usaha["Status Kelayakan"], df_usaha["Rekomendasi Investasi (Rp)"] = zip(*[
        get_status(score, modal)
        for score, modal in zip(df_usaha["Skor CODAS"], df_kriteria["Modal Awal (Rp)"])
    ])

    # Tampilkan & unduh hasil
    st.subheader(labels[lang]['hasil'])
    hasil = df_usaha[[
        labels[lang]['nama_usaha'],
        "Skor CODAS",
        "Peringkat",
        "Status Kelayakan",
        "Rekomendasi Investasi (Rp)"
    ]].sort_values("Peringkat").reset_index(drop=True)
    st.dataframe(
        hasil.style.format({
            "Skor CODAS": "{:.4f}",
            "Rekomendasi Investasi (Rp)": "Rp {:,.0f}"
        }),
        use_container_width=True
    )
    st.download_button(
        labels[lang]['download_hasil'],
        data=hasil.to_csv(index=False),
        file_name="hasil_investasi.csv",
        mime="text/csv"
    )
