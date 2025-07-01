import streamlit as st
import pandas as pd
import numpy as np

# === PAGE CONFIG ===
st.set_page_config(page_title="SPK Investasi Mahasiswa", layout="wide")

# === CUSTOM STYLE ===
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #ffffff;
        color: #1F1F1F;
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
        background-color: #00BFFF;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }

    h1, h2, h3, h4 {
        color: #1F1F1F;
    }

    .dataframe th {
        background-color: #F0F8FF;
        color: #1F1F1F;
    }
    .dataframe td {
        text-align: center;
        padding: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.image("https://img.icons8.com/ios-filled/50/database.png", width=30)
st.sidebar.title("SPK Investasi Mahasiswa")
st.sidebar.markdown("### Metode Input Data")

# Tombol vertikal
manual_click = st.sidebar.button("📝 Input Manual", key="btn_manual")
upload_click = st.sidebar.button("📁 Upload File", key="btn_upload")

# Handle session state
if 'input_method' not in st.session_state:
    st.session_state.input_method = "Manual"

if manual_click:
    st.session_state.input_method = "Manual"
if upload_click:
    st.session_state.input_method = "Upload"

input_method = st.session_state.input_method

# === KONSTAN ===
kriteria_cols = ['ROI (%)', 'Modal Awal (Rp)', 'Pendapatan Rata-Rata 3 Bulan (Rp)',
                 'Aset (Rp)', 'Inovasi Produk (1-5)', 'Peluang Pasar (1-5)',
                 'Tingkat Risiko (1-5)']
cost_indices = [1, 6]

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
st.title("📊 Sistem Pendukung Keputusan Investasi Usaha Mahasiswa")

df_usaha = None

# === INPUT MANUAL ===
if input_method == "Manual":
    st.subheader("📝 Input Manual")
    num = st.number_input("Jumlah Usaha", min_value=1, max_value=20, step=1)
    default_data = pd.DataFrame({
        "Nama Usaha": [f"Usaha {i+1}" for i in range(num)],
        **{col: [0.0]*num for col in kriteria_cols}
    })
    df_input = st.data_editor(default_data, use_container_width=True, num_rows="dynamic")
    if st.button("💾 Simpan & Tampilkan Hasil", key="process_manual"):
        df_usaha = df_input.copy()

# === INPUT FILE ===
elif input_method == "Upload":
    st.subheader("📁 Upload File")
    uploaded_file = st.file_uploader("Unggah file CSV/XLSX", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_usaha = pd.read_csv(uploaded_file)
            else:
                df_usaha = pd.read_excel(uploaded_file)
            st.success("✅ Data berhasil dimuat!")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

# === PROSES & HASIL ===
if df_usaha is not None:
    st.subheader("📄 Data Usaha Mahasiswa")
    st.dataframe(df_usaha.reset_index(drop=True), use_container_width=True)

    df_kriteria = df_usaha[kriteria_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    weights, data_normalized = calculate_critic(df_kriteria, cost_indices)

    st.subheader("📌 Bobot Kriteria (Metode CRITIC)")
    st.write(weights)

    df_usaha['Skor CODAS'] = calculate_codas(data_normalized, weights)

    # FIXED NaN to int issue
    df_usaha['Peringkat'] = df_usaha['Skor CODAS'].rank(ascending=False, method='min').fillna(0).astype(int)

    df_usaha['Status Kelayakan'], df_usaha['Rekomendasi Investasi (Rp)'] = zip(
        *df_usaha['Skor CODAS'].apply(get_status_and_recommendation))

    st.subheader("📈 Hasil Rekomendasi Investasi")
    df_output = df_usaha[['Peringkat', 'Nama Usaha', 'Skor CODAS', 'Status Kelayakan', 'Rekomendasi Investasi (Rp)']]
    df_output = df_output.sort_values(by='Peringkat').reset_index(drop=True)

    st.dataframe(
        df_output.style.format({
            "Skor CODAS": "{:.4f}",
            "Rekomendasi Investasi (Rp)": "Rp {:,.0f}"
        }).set_properties({
            'text-align': 'center'
        }).set_properties(subset=['Nama Usaha'], **{
            'text-align': 'left'
        }),
        use_container_width=True
    )

    csv = df_output.to_csv(index=False)
    st.download_button("💾 Unduh Hasil", data=csv, file_name="hasil_investasi.csv", mime="text/csv")
