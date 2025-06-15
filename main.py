
import numpy as np
import pandas as pd

def calculate_critic(df, cost_cols=[]):
    # Step 1: Normalisasi (benefit / cost)
    norm = df.copy()
    for col in df.columns:
        if col in cost_cols:
            norm[col] = df[col].min() / df[col]
        else:
            norm[col] = df[col] / df[col].max()

    # Step 2: Standar deviasi
    std_dev = norm.std()

    # Step 3: Korelasi absolut
    corr = norm.corr().abs()

    # Step 4: Konflik informasi
    conflict = 1 - corr
    info = std_dev * conflict.sum()

    # Step 5: Bobot
    weights = info / info.sum()
    return weights, norm

def calculate_codas(df_normalized, weights, tau=0.02):
    # Step 1: Matriks r_ij
    weighted = df_normalized * weights

    # Step 2: Solusi ideal negatif
    s_j = weighted.min()

    # Step 3: Euclidean & Taxicab untuk semua alternatif
    E = ((weighted - s_j) ** 2).sum(axis=1).pow(0.5)
    T = (weighted - s_j).abs().sum(axis=1)
    H = E + tau * T

    # Step 4: Normalisasi skor 0–1
    H_norm = (H - H.min()) / (H.max() - H.min())
    return H_norm


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
    }
}

standard_kriteria = labels['id']['kriteria']

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

def get_status_and_recommendation(score, modal_awal):
    stts = labels[lang]['status']
    if score >= 0.81:
        return stts['sangat_layak'], modal_awal * 0.60
    elif score >= 0.61:
        return stts['layak'], modal_awal * 0.45
    elif score >= 0.41:
        return stts['cukup_layak'], modal_awal * 0.30
    elif score >= 0.21:
        return stts['kurang_layak'], modal_awal * 0.15
    else:
        return stts['tidak_layak'], 0.0

# === MAIN ===
st.title(labels[lang]['title'])
df_usaha = None

if input_method == "Manual":
    st.subheader(labels[lang]['manual'])
    num = st.number_input(labels[lang]['num_usaha'], min_value=1, max_value=20, step=1)
    default_data = pd.DataFrame({
        labels[lang]['nama_usaha']: [f"Business {i+1}" if lang == 'en' else f"Usaha {i+1}" for i in range(num)],
        **{col: [0.0]*num for col in labels[lang]['kriteria']}
    })
    df_input = st.data_editor(default_data, use_container_width=True, num_rows="dynamic")
    if st.button(labels[lang]['save'], key="process_manual"):
        df_usaha = df_input.copy()

elif input_method == "Upload":
    st.subheader(labels[lang]['upload'])
    template_df = pd.DataFrame({
        labels[lang]['nama_usaha']: [""],
        **{col: [0.0] for col in labels[lang]['kriteria']}
    })
    template_csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=labels[lang]['download_template'],
        data=template_csv,
        file_name='template_input_usaha.csv',
        mime='text/csv'
    )
    uploaded_file = st.file_uploader(labels[lang]['upload_prompt'], type=["csv"])
    if uploaded_file is not None:
        try:
            df_usaha = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            df_usaha = None

if df_usaha is not None:
    st.subheader(labels[lang]['data_usaha'])
    st.dataframe(df_usaha.reset_index(drop=True), use_container_width=True)

    df_kriteria = df_usaha[labels[lang]['kriteria']].apply(pd.to_numeric, errors='coerce').fillna(0)
    weights, df_normalized = calculate_critic(df_kriteria, cost_cols=["Modal Awal (Rp)", "Tingkat Risiko (1-5)"])

    st.subheader(labels[lang]['bobot'])
    st.write(weights)

    df_usaha['Skor CODAS'] = calculate_codas(df_normalized, weights)
    df_usaha['Peringkat'] = df_usaha['Skor CODAS'].rank(ascending=False, method='min').astype(int)
    df_usaha['Status Kelayakan'], df_usaha['Rekomendasi Investasi (Rp)'] = zip(
        *[get_status_and_recommendation(score, modal) for score, modal in zip(df_usaha['Skor CODAS'], df_kriteria['Modal Awal (Rp)'])]
    )

    st.subheader(labels[lang]['hasil'])
    df_usaha[labels[lang]['nama_usaha']] = df_usaha[labels[lang]['nama_usaha']].fillna("-")
    df_output = df_usaha[['Peringkat', labels[lang]['nama_usaha'], 'Skor CODAS', 'Status Kelayakan', 'Rekomendasi Investasi (Rp)']]
    df_output = df_output.sort_values(by='Peringkat').reset_index(drop=True)

    st.dataframe(
        df_output.style.format({
            "Skor CODAS": "{:.4f}",
            "Rekomendasi Investasi (Rp)": "Rp {:,.0f}"
        }).set_properties(**{'text-align': 'center'}).set_properties(
            subset=[labels[lang]['nama_usaha']], **{'text-align': 'left'}),
        use_container_width=True
    )

    csv = df_output.to_csv(index=False)
    st.download_button(labels[lang]['download_hasil'], data=csv, file_name="hasil_investasi.csv", mime="text/csv")
