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
    },
    'en': {
        'title': "📊 Decision Support System for Student Business Investment",
        'manual': "📝 Manual Input",
        'upload': "📁 Upload File",
        'num_usaha': "Number of Businesses",
        'save': "💾 Save & Show Result",
        'download_template': "⬇️ Download Blank Template (CSV)",
        'upload_prompt': "Upload CSV file",
        'data_usaha': "📄 Student Business Data",
        'bobot': "📌 Criteria Weights (CRITIC Method)",
        'hasil': "📈 Investment Recommendation Result",
        'download_hasil': "💾 Download Result",
        'change_lang': "🇮🇩 Bahasa Indonesia",
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

# === FUNGSI ===
def calculate_critic(data, cost_indices=[]):
    data_norm = data.copy()
    for i, col in enumerate(data.columns):
        if i in cost_indices:
            data_norm[col] = data[col].min() / data[col]
        else:
            data_norm[col] = data[col] / data[col].max()
    std_dev = data_norm.std()
    corr_matrix = data_norm.corr()
    conflict = 1 - corr_matrix.abs()
    info = std_dev * conflict.sum()
    weights = info / info.sum()
    return weights, data_norm

def calculate_codas(data, weights, cost_indices=[]):
    data_norm = data.copy()
    for i, col in enumerate(data.columns):
        if i in cost_indices:
            data_norm[col] = data[col].min() / data[col]
        else:
            data_norm[col] = data[col] / data[col].max()
    weighted = data_norm * weights.values
    negative_ideal = weighted.min()
    euclidean = np.sqrt(((weighted - negative_ideal) ** 2).sum(axis=1))
    taxicab = np.abs(weighted - negative_ideal).sum(axis=1)
    score = euclidean + taxicab
    return (score - score.min()) / (score.max() - score.min())

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
    st.download_button(
        label=labels[lang]['download_template'],
        data=template_df.to_csv(index=False).encode('utf-8'),
        file_name='template_input_usaha.csv',
        mime='text/csv'
    )
    uploaded_file = st.file_uploader(labels[lang]['upload_prompt'], type=["csv"])
    if uploaded_file is not None:
        try:
            df_usaha = pd.read_csv(uploaded_file)
            missing = set(labels[lang]['kriteria']) - set(df_usaha.columns)
            if missing:
                st.error("❌ Kolom berikut tidak ditemukan: " + ", ".join(missing))
                df_usaha = None
            else:
                st.success("✅ Data berhasil dimuat!" if lang == 'id' else "✅ File loaded successfully!")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}" if lang == 'id' else f"Failed to read file: {e}")

if df_usaha is not None:
    st.subheader(labels[lang]['data_usaha'])
    st.dataframe(df_usaha.reset_index(drop=True), use_container_width=True)

    col_map = dict(zip(labels[lang]['kriteria'], standard_kriteria))
    df_kriteria = df_usaha.rename(columns=col_map)[standard_kriteria].apply(pd.to_numeric, errors='coerce').fillna(0)

    weights, _ = calculate_critic(df_kriteria, cost_indices=[1, 6])

    st.subheader(labels[lang]['bobot'])
    st.write(weights)

    df_usaha['Skor CODAS'] = calculate_codas(df_kriteria, weights, cost_indices=[1, 6])
    df_usaha['Peringkat'] = df_usaha['Skor CODAS'].rank(ascending=False, method='min').astype(int)
    df_usaha['Status Kelayakan'], df_usaha['Rekomendasi Investasi (Rp)'] = zip(
        *[get_status_and_recommendation(score, modal) for score, modal in zip(df_usaha['Skor CODAS'], df_kriteria['Modal Awal (Rp)'])])

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

    st.download_button(labels[lang]['download_hasil'], data=df_output.to_csv(index=False), file_name="hasil_investasi.csv", mime="text/csv")
