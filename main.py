import streamlit as st
import pandas as pd
import numpy as np

# === PAGE CONFIG ===
st.set_page_config(page_title="SPK Investasi Mahasiswa", layout="wide")

# === SESSION STATE ===
if 'language' not in st.session_state:
    st.session_state.language = 'id'

def switch_language():
    st.session_state.language = 'en' if st.session_state.language == 'id' else 'id'

# === TEXT MULTI-BAHASA ===
TEXTS = {
    "id": {
        "title": "📊 Sistem Pendukung Keputusan Investasi Usaha Mahasiswa",
        "sidebar_title": "SPK Investasi Mahasiswa",
        "input_method": "### Metode Input Data",
        "manual_input": "📝 Input Manual",
        "upload_file": "📁 Upload File",
        "jumlah_usaha": "Jumlah Usaha",
        "save_and_process": "💾 Simpan & Tampilkan Hasil",
        "upload_section": "📁 Upload File",
        "download_template": "⬇️ Unduh Template Kosong (CSV)",
        "upload_prompt": "Unggah file CSV/XLSX",
        "success_upload": "✅ Data berhasil dimuat!",
        "failed_upload": "Gagal membaca file: ",
        "data_title": "📄 Data Usaha Mahasiswa",
        "weights_title": "📌 Bobot Kriteria (Metode CRITIC)",
        "results_title": "📈 Hasil Rekomendasi Investasi",
        "download_result": "💾 Unduh Hasil",
        "switch_lang": "🔁 Ganti ke Bahasa Inggris"
    },
    "en": {
        "title": "📊 Decision Support System for Student Business Investment",
        "sidebar_title": "Student Investment DSS",
        "input_method": "### Data Input Method",
        "manual_input": "📝 Manual Input",
        "upload_file": "📁 Upload File",
        "jumlah_usaha": "Number of Businesses",
        "save_and_process": "💾 Save & Show Results",
        "upload_section": "📁 Upload File",
        "download_template": "⬇️ Download Blank Template (CSV)",
        "upload_prompt": "Upload CSV/XLSX file",
        "success_upload": "✅ Data loaded successfully!",
        "failed_upload": "Failed to read file: ",
        "data_title": "📄 Student Business Data",
        "weights_title": "📌 Criteria Weights (CRITIC Method)",
        "results_title": "📈 Investment Recommendation Results",
        "download_result": "💾 Download Results",
        "switch_lang": "🔁 Switch to Bahasa Indonesia"
    }
}

KRITERIA_LABELS = {
    "id": {
        'ROI (%)': 'ROI (%)',
        'Modal Awal (Rp)': 'Modal Awal (Rp)',
        'Pendapatan Rata-Rata 3 Bulan (Rp)': 'Pendapatan Rata-Rata 3 Bulan (Rp)',
        'Aset (Rp)': 'Aset (Rp)',
        'Inovasi Produk (1-5)': 'Inovasi Produk (1-5)',
        'Peluang Pasar (1-5)': 'Peluang Pasar (1-5)',
        'Tingkat Risiko (1-5)': 'Tingkat Risiko (1-5)'
    },
    "en": {
        'ROI (%)': 'ROI (%)',
        'Modal Awal (Rp)': 'Initial Capital (Rp)',
        'Pendapatan Rata-Rata 3 Bulan (Rp)': 'Avg. 3-Month Revenue (Rp)',
        'Aset (Rp)': 'Assets (Rp)',
        'Inovasi Produk (1-5)': 'Product Innovation (1-5)',
        'Peluang Pasar (1-5)': 'Market Opportunity (1-5)',
        'Tingkat Risiko (1-5)': 'Risk Level (1-5)'
    }
}

# === STYLE ===
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
lang = st.session_state.language
st.sidebar.title(TEXTS[lang]["sidebar_title"])
st.sidebar.markdown(TEXTS[lang]["input_method"])
st.sidebar.button(TEXTS[lang]["switch_lang"], on_click=switch_language)
manual_click = st.sidebar.button(TEXTS[lang]["manual_input"], key="btn_manual")
upload_click = st.sidebar.button(TEXTS[lang]["upload_file"], key="btn_upload")

# === KONSTAN ===
kriteria_cols = list(KRITERIA_LABELS["id"].keys())
cost_indices = [1, 6]

# === SESSION ===
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

def get_status_and_recommendation(score):
    if score >= 0.81:
        return "Sangat Layak", 20000000
    elif score >= 0.61:
        return "Layak", 15000000
    elif score >= 0.41:
        return "Cukup Layak", 10000000
    elif score >= 0.21:
        return "Kurang Layak", 5000000
    else:
        return "Tidak Layak", 0

# === MAIN ===
st.title(TEXTS[lang]["title"])
df_usaha = None

if input_method == "Manual":
    st.subheader(TEXTS[lang]["manual_input"])
    num = st.number_input(TEXTS[lang]["jumlah_usaha"], min_value=1, max_value=20, step=1)
    default_data = pd.DataFrame({
        "Nama Usaha": [f"Usaha {i+1}" for i in range(num)],
        **{KRITERIA_LABELS[lang][col]: [0.0]*num for col in kriteria_cols}
    })
    df_input = st.data_editor(default_data, use_container_width=True, num_rows="dynamic")
    if st.button(TEXTS[lang]["save_and_process"], key="process_manual"):
        df_input.columns = ["Nama Usaha"] + kriteria_cols  # normalize column names back
        df_usaha = df_input.copy()

elif input_method == "Upload":
    st.subheader(TEXTS[lang]["upload_section"])
    template_df = pd.DataFrame({
        "Nama Usaha": [""],
        **{KRITERIA_LABELS[lang][col]: [0.0] for col in kriteria_cols}
    })
    template_df.columns = ["Nama Usaha"] + kriteria_cols
    template_csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(TEXTS[lang]["download_template"], data=template_csv, file_name="template_input_usaha.csv", mime="text/csv")
    uploaded_file = st.file_uploader(TEXTS[lang]["upload_prompt"], type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_usaha = pd.read_csv(uploaded_file)
            else:
                df_usaha = pd.read_excel(uploaded_file)
            st.success(TEXTS[lang]["success_upload"])
        except Exception as e:
            st.error(TEXTS[lang]["failed_upload"] + str(e))

# === OUTPUT ===
if df_usaha is not None:
    st.subheader(TEXTS[lang]["data_title"])
    st.dataframe(df_usaha.reset_index(drop=True), use_container_width=True)

    df_kriteria = df_usaha[kriteria_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    weights, data_normalized = calculate_critic(df_kriteria, cost_indices)

    st.subheader(TEXTS[lang]["weights_title"])
    st.write(weights)

    df_usaha['Skor CODAS'] = calculate_codas(data_normalized, weights)
    df_usaha['Peringkat'] = df_usaha['Skor CODAS'].rank(ascending=False, method='min').fillna(0).astype(int)
    df_usaha['Status Kelayakan'], df_usaha['Rekomendasi Investasi (Rp)'] = zip(
        *df_usaha['Skor CODAS'].apply(get_status_and_recommendation))

    st.subheader(TEXTS[lang]["results_title"])
    df_output = df_usaha[['Peringkat', 'Nama Usaha', 'Skor CODAS', 'Status Kelayakan', 'Rekomendasi Investasi (Rp)']]
    df_output = df_output.sort_values(by='Peringkat').reset_index(drop=True)

    st.dataframe(
        df_output.style.format({
            "Skor CODAS": "{:.4f}",
            "Rekomendasi Investasi (Rp)": "Rp {:,.0f}"
        }).set_properties(**{
            'text-align': 'center'
        }).set_properties(subset=['Nama Usaha'], **{
            'text-align': 'left'
        }),
        use_container_width=True
    )

    csv = df_output.to_csv(index=False)
    st.download_button(TEXTS[lang]["download_result"], data=csv, file_name="hasil_investasi.csv", mime="text/csv")
