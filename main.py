import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SPK Investasi Mahasiswa", layout="wide")

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
    </style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.title("SPK Investasi Mahasiswa")
st.sidebar.markdown("### Metode Input Data")
manual_click = st.sidebar.button("📝 Input Manual", key="btn_manual")
upload_click = st.sidebar.button("📁 Upload File", key="btn_upload")

# === KONSTAN ===
kriteria_cols = ['ROI (%)', 'Modal Awal (Rp)', 'Pendapatan Rata-Rata 3 Bulan (Rp)',
                 'Aset (Rp)', 'Inovasi Produk (1-5)', 'Peluang Pasar (1-5)', 'Tingkat Risiko (1-5)']
cost_indices = [1, 6]  # indeks cost

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

def get_status_and_recommendation(score, modal_awal):
    if score >= 0.81:
        return "Sangat Layak", modal_awal * 0.60
    elif score >= 0.61:
        return "Layak", modal_awal * 0.30
    elif score >= 0.41:
        return "Cukup Layak", modal_awal * 0.45
    elif score >= 0.21:
        return "Kurang Layak", modal_awal * 0.15
    else:
        return "Tidak Layak", 0.0

# === MAIN ===
st.title("📊 Sistem Pendukung Keputusan Investasi Usaha Mahasiswa")
df_usaha = None

if input_method == "Manual":
    st.subheader("📝 Input Manual (Ramah Mobile)")
    num = st.number_input("Jumlah Usaha", min_value=1, max_value=10, step=1)
    usaha_list = []

    for i in range(num):
        with st.expander(f"➕ Formulir Usaha {i+1}", expanded=True):
            nama = st.text_input(f"Nama Usaha", key=f"nama_{i}")
            roi = st.number_input(f"ROI (%)", min_value=0.0, step=0.1, key=f"roi_{i}")
            modal = st.number_input(f"Modal Awal (Rp)", min_value=0.0, step=1000.0, key=f"modal_{i}")
            pendapatan = st.number_input(f"Pendapatan Rata-Rata 3 Bulan (Rp)", min_value=0.0, step=1000.0, key=f"pend_{i}")
            aset = st.number_input(f"Aset (Rp)", min_value=0.0, step=1000.0, key=f"aset_{i}")
            inovasi = st.slider(f"Inovasi Produk (1-5)", 1, 5, key=f"inov_{i}")
            peluang = st.slider(f"Peluang Pasar (1-5)", 1, 5, key=f"peluang_{i}")
            risiko = st.slider(f"Tingkat Risiko (1-5)", 1, 5, key=f"risiko_{i}")

            usaha_list.append({
                "Nama Usaha": nama,
                "ROI (%)": roi,
                "Modal Awal (Rp)": modal,
                "Pendapatan Rata-Rata 3 Bulan (Rp)": pendapatan,
                "Aset (Rp)": aset,
                "Inovasi Produk (1-5)": inovasi,
                "Peluang Pasar (1-5)": peluang,
                "Tingkat Risiko (1-5)": risiko,
            })

    if st.button("💾 Simpan & Tampilkan Hasil", key="process_manual"):
        df_usaha = pd.DataFrame(usaha_list)

elif input_method == "Upload":
    st.subheader("📁 Upload File")
    template_df = pd.DataFrame({
        "Nama Usaha": [""],
        **{col: [0.0] for col in kriteria_cols}
    })
    template_csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Unduh Template Kosong (CSV)",
        data=template_csv,
        file_name='template_input_usaha.csv',
        mime='text/csv'
    )
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

# === PROSES & OUTPUT ===
if df_usaha is not None:
    st.subheader("📄 Data Usaha Mahasiswa")
    st.dataframe(df_usaha.reset_index(drop=True), use_container_width=True)

    df_kriteria = df_usaha[kriteria_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    weights, data_normalized = calculate_critic(df_kriteria, cost_indices)

    st.subheader("📌 Bobot Kriteria (Metode CRITIC)")
    st.write(weights)

    df_usaha['Skor CODAS'] = calculate_codas(data_normalized, weights)
    df_usaha['Peringkat'] = df_usaha['Skor CODAS'].rank(ascending=False, method='min').fillna(0).astype(int)
    df_usaha['Status Kelayakan'], df_usaha['Rekomendasi Investasi (Rp)'] = zip(
        *[get_status_and_recommendation(score, modal) for score, modal in zip(df_usaha['Skor CODAS'], df_kriteria['Modal Awal (Rp)'])])

    st.subheader("📈 Hasil Rekomendasi Investasi")
    df_output = df_usaha[['Peringkat', 'Nama Usaha', 'Skor CODAS', 'Status Kelayakan', 'Rekomendasi Investasi (Rp)']]
    df_output = df_output.sort_values(by='Peringkat').reset_index(drop=True)

    st.dataframe(
        df_output.style.format({
            "Skor CODAS": "{:.4f}",
            "Rekomendasi Investasi (Rp)": "Rp {:,.0f}"
        }).set_properties(**{'text-align': 'center'}).set_properties(
            subset=['Nama Usaha'], **{'text-align': 'left'}),
        use_container_width=True
    )

    csv = df_output.to_csv(index=False)
    st.download_button("💾 Unduh Hasil", data=csv, file_name="hasil_investasi.csv", mime="text/csv")
