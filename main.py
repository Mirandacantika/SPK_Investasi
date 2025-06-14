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

# === FUNGSI ===
def calculate_critic(data, cost_indices=[]):
    data_normalized = data.copy()
    for i, col in enumerate(data.columns):
        if i in cost_indices:
            data_normalized[col] = data[col].min() / data[col]
        else:
            data_normalized[col] = data[col] / data[col].max()
    mean = data_normalized.mean()
    std_dev = ((data_normalized - mean) ** 2).mean() ** 0.5
    corr_matrix = data_normalized.corr()
    conflict = 1 - corr_matrix.abs()
    info = std_dev * conflict.sum()
    weights = info / info.sum()
    return weights, data_normalized

def calculate_codas(data_normalized, weights):
    weighted_data = data_normalized * weights.values
    ideal_solution = weighted_data.min()
    E = np.sqrt(((weighted_data - ideal_solution) ** 2).sum(axis=1))
    T = (weighted_data - ideal_solution).abs().sum(axis=1)
    n = len(E)
    tau = 0.01
    H_matrix = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            delta_e = E[i] - E[k]
            delta_t = T[i] - T[k]
            mu = 1 if abs(delta_e) >= tau else 0
            H_matrix[i, k] = delta_e + mu * delta_t
    H_scores = H_matrix.sum(axis=1)
    return H_scores

def get_status_and_recommendation(score, modal):
    if score > 0.80:
        return labels[lang]['status']['sangat_layak'], modal * 0.60
    elif score > 0.60:
        return labels[lang]['status']['layak'], modal * 0.45
    elif score > 0.40:
        return labels[lang]['status']['cukup_layak'], modal * 0.30
    elif score > 0.20:
        return labels[lang]['status']['kurang_layak'], modal * 0.15
    else:
        return labels[lang]['status']['tidak_layak'], 0.0

# === ANTARMUKA ===
st.title(labels[lang]['title'])

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

st.subheader(labels[lang][input_method.lower()])
df_usaha = None

if input_method == "Manual":
    num = st.number_input(labels[lang]['num_usaha'], min_value=1, max_value=20, step=1)
    default_data = pd.DataFrame({
        labels[lang]['nama_usaha']: [f"Usaha {i+1}" for i in range(num)],
        **{col: [0.0]*num for col in labels[lang]['kriteria']}
    })
    df_input = st.data_editor(default_data, use_container_width=True, num_rows="dynamic")
    if st.button(labels[lang]['save']):
        df_usaha = df_input.copy()
else:
    template_df = pd.DataFrame({labels[lang]['nama_usaha']: [""], **{col: [0.0] for col in labels[lang]['kriteria']}})
    st.download_button(labels[lang]['download_template'], data=template_df.to_csv(index=False), file_name='template_usaha.csv', mime='text/csv')
    file = st.file_uploader(labels[lang]['upload_prompt'], type=["csv"])
    if file:
        df_usaha = pd.read_csv(file)

if df_usaha is not None:
    st.subheader(labels[lang]['data_usaha'])
    st.dataframe(df_usaha)

    col_map = dict(zip(labels[lang]['kriteria'], standard_kriteria))
    df_kriteria = df_usaha.rename(columns=col_map)[standard_kriteria].apply(pd.to_numeric, errors='coerce').fillna(0)

    weights, data_normalized = calculate_critic(df_kriteria, cost_indices=[1, 6])
    st.subheader(labels[lang]['bobot'])
    st.write(weights)

    scores = calculate_codas(data_normalized, weights)
    df_usaha['Skor CODAS'] = scores
    df_usaha['Peringkat'] = df_usaha['Skor CODAS'].rank(ascending=False, method='min').astype(int)
    df_usaha['Status Kelayakan'], df_usaha['Rekomendasi Investasi (Rp)'] = zip(*[
        get_status_and_recommendation(score, modal) for score, modal in zip(df_usaha['Skor CODAS'], df_kriteria['Modal Awal (Rp)'])
    ])

    st.subheader(labels[lang]['hasil'])
    df_output = df_usaha[[labels[lang]['nama_usaha'], 'Peringkat', 'Skor CODAS', 'Status Kelayakan', 'Rekomendasi Investasi (Rp)']].sort_values(by='Peringkat')
    st.dataframe(df_output.style.format({"Skor CODAS": "{:.4f}", "Rekomendasi Investasi (Rp)": "Rp {:,.0f}"}), use_container_width=True)

    st.download_button(labels[lang]['download_hasil'], data=df_output.to_csv(index=False), file_name='hasil_investasi.csv', mime='text/csv')
