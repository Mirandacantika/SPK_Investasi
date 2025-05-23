import streamlit as st
import pandas as pd
import numpy as np

# === PAGE CONFIG ===
st.set_page_config(page_title="SPK Investasi Mahasiswa", layout="wide")

# === BAHASA ===
if 'language' not in st.session_state:
    st.session_state.language = 'id'

if st.sidebar.button("🇮🇩 Bahasa Indonesia"):
    st.session_state.language = 'id'
if st.sidebar.button("🇬🇧 English"):
    st.session_state.language = 'en'

lang = st.session_state.language

labels = {
    'id': {
        'title': "📊 Sistem Pendukung Keputusan Investasi Usaha Mahasiswa",
        'manual': "📝 Input Manual",
        'upload': "📁 Upload File",
        'num_usaha': "Jumlah Usaha",
        'save': "💾 Simpan & Tampilkan Hasil",
        'download_template': "⬇️ Unduh Template Kosong (CSV)",
        'upload_prompt': "Unggah file CSV/XLSX",
        'data_usaha': "📄 Data Usaha Mahasiswa",
        'bobot': "📌 Bobot Kriteria (Metode CRITIC)",
        'hasil': "📈 Hasil Rekomendasi Investasi",
        'download_hasil': "💾 Unduh Hasil",
        'kriteria': ['ROI (%)', 'Modal Awal (Rp)', 'Pendapatan Rata-Rata 3 Bulan (Rp)', 'Aset (Rp)',
                     'Inovasi Produk (1-5)', 'Peluang Pasar (1-5)', 'Tingkat Risiko (1-5)'],
        'status': {
            'sangat_layak': "Sangat Layak",
            'layak': "Layak",
            'cukup_layak': "Cukup Layak",
            'kurang_layak': "Kurang Layak",
            'tidak_layak': "Tidak Layak"
        }
    },
    'en': {
        'title': "📊 Decision Support System for Student Business Investment",
        'manual': "📝 Manual Input",
        'upload': "📁 Upload File",
        'num_usaha': "Number of Businesses",
        'save': "💾 Save & Show Result",
        'download_template': "⬇️ Download Blank Template (CSV)",
        'upload_prompt': "Upload CSV/XLSX file",
        'data_usaha': "📄 Student Business Data",
        'bobot': "📌 Criteria Weights (CRITIC Method)",
        'hasil': "📈 Investment Recommendation Result",
        'download_hasil': "💾 Download Result",
        'kriteria': ['ROI (%)', 'Initial Capital (Rp)', 'Avg. 3-Month Revenue (Rp)', 'Assets (Rp)',
                     'Product Innovation (1-5)', 'Market Opportunity (1-5)', 'Risk Level (1-5)'],
        'status': {
            'sangat_layak': "Highly Recommended",
            'layak': "Recommended",
            'cukup_layak': "Moderately Recommended",
            'kurang_layak': "Less Recommended",
            'tidak_layak': "Not Recommended"
        }
    }
}

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
manual_click = st.sidebar.button(labels[lang]['manual'], key="btn_manual")
upload_click = st.sidebar.button(labels[lang]['upload'], key="btn_upload")

# === KONSTAN ===
kriteria_cols = labels[lang]['kriteria']
cost_indices = [1, 6]  # indeks kriteria cost

if 'input_method' not in st.session_state:
    st.session_state.input_method = "Manual"
if manual_click:
    st.session_state.input_method = "Manual"
if upload_click:
    st.session_state.input_method = "Upload"
input_method = st.session_state.input_method

# === FUNGSI ===
def calculate_critic(data, cost_indices=[]):
    data_normalized = data.copy()
    for i, col in enumerate(data.columns):
        if i in cost_indices:
            data_normalized[col] = data[col].min() / data[col]
        else:
            data_normalized[col] = data[col] / data[col].max()
    std_dev = data_normalized.std()
    corr_matrix = data_normalized.corr()
    conflict = 1 - corr_matrix.abs()
    info = std_dev * conflict.sum()
    weights = info / info.sum()
    return weights, data_normalized

def calculate_codas(data_normalized, weights):
    weighted_data = data_normalized * weights.values
    ideal_solution = weighted_data.min()
    euclidean = np.sqrt(((weighted_data - ideal_solution) ** 2).sum(axis=1))
    taxicab = np.abs(weighted_data - ideal_solution).sum(axis=1)
    score = euclidean + taxicab
    score_normalized = (score - score.min()) / (score.max() - score.min())
    return score_normalized

def get_status_and_recommendation(score, modal_awal):
    stts = labels[lang]['status']
    if score >= 0.81:
        return stts['sangat_layak'], modal_awal * 0.60
    elif score >= 0.61:
        return stts['layak'], modal_awal * 0.30
    elif score >= 0.41:
        return stts['cukup_layak'], modal_awal * 0.45
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
        "Business Name" if lang == 'en' else "Nama Usaha": [f"Business {i+1}" if lang == 'en' else f"Usaha {i+1}" for i in range(num)],
        **{col: [0.0]*num for col in kriteria_cols}
    })
    df_input = st.data_editor(default_data, use_container_width=True, num_rows="dynamic")
    if st.button(labels[lang]['save'], key="process_manual"):
        df_usaha = df_input.copy()

elif input_method == "Upload":
    st.subheader(labels[lang]['upload'])
    template_df = pd.DataFrame({
        "Business Name" if lang == 'en' else "Nama Usaha": [""],
        **{col: [0.0] for col in kriteria_cols}
    })
    template_csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=labels[lang]['download_template'],
        data=template_csv,
        file_name='template_input_usaha.csv',
        mime='text/csv'
    )
    uploaded_file = st.file_uploader(labels[lang]['upload_prompt'], type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_usaha = pd.read_csv(uploaded_file)
            else:
                df_usaha = pd.read_excel(uploaded_file)
            st.success("✅ File loaded successfully!" if lang == 'en' else "✅ Data berhasil dimuat!")
        except Exception as e:
            st.error(f"Failed to read file: {e}" if lang == 'en' else f"Gagal membaca file: {e}")

if df_usaha is not None:
    st.subheader(labels[lang]['data_usaha'])
    st.dataframe(df_usaha.reset_index(drop=True), use_container_width=True)

    df_kriteria = df_usaha[kriteria_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    weights, data_normalized = calculate_critic(df_kriteria, cost_indices)

    st.subheader(labels[lang]['bobot'])
    st.write(weights)

    df_usaha['Skor CODAS'] = calculate_codas(data_normalized, weights)
    df_usaha['Peringkat'] = df_usaha['Skor CODAS'].rank(ascending=False, method='min').fillna(0).astype(int)
    df_usaha['Status Kelayakan'], df_usaha['Rekomendasi Investasi (Rp)'] = zip(
        *[get_status_and_recommendation(score, modal) for score, modal in zip(df_usaha['Skor CODAS'], df_kriteria[kriteria_cols[1]])])

    st.subheader(labels[lang]['hasil'])
    df_output = df_usaha[['Peringkat', 'Nama Usaha' if lang == 'id' else 'Business Name', 'Skor CODAS', 'Status Kelayakan', 'Rekomendasi Investasi (Rp)']]
    df_output = df_output.sort_values(by='Peringkat').reset_index(drop=True)

    st.dataframe(
        df_output.style.format({
            "Skor CODAS": "{:.4f}",
            "Rekomendasi Investasi (Rp)": "Rp {:,.0f}"
        }).set_properties(**{'text-align': 'center'}).set_properties(
            subset=['Nama Usaha' if lang == 'id' else 'Business Name'], **{'text-align': 'left'}),
        use_container_width=True
    )

    csv = df_output.to_csv(index=False)
    st.download_button(labels[lang]['download_hasil'], data=csv, file_name="hasil_investasi.csv", mime="text/csv")
