import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# === PAGE CONFIG ===
st.set_page_config(page_title="SPK Investasi Mahasiswa", layout="wide")

# === BAHASA & LABELS ===
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
def calculate_critic(df, cost_cols=[]):
    norm = df.copy()
    for c in df.columns:
        if c in cost_cols:
            # hindari pembagian 0/0
            norm[c] = np.where(df[c] != 0, df[c].min() / df[c], 0)
        else:
            norm[c] = np.where(df[c].max() != 0, df[c] / df[c].max(), 0)
    std = norm.std()
    corr = norm.corr()
    conflict = 1 - corr.abs()
    info = std * conflict.sum()
    weights = info / info.sum()
    return weights, norm

def euclid(a, b):
    return np.linalg.norm(a - b)

def taxi(a, b):
    return np.sum(np.abs(a - b))

def calculate_codas_full(norm_df, weights):
    W = norm_df * weights
    ideal_neg = W.min()
    e_dist = W.apply(lambda row: euclid(row.values, ideal_neg.values), axis=1)
    t_dist = W.apply(lambda row: taxi(row.values, ideal_neg.values), axis=1)
    mu = 0.02
    scores = []
    for i in range(len(W)):
        ra = 0
        for j in range(len(W)):
            if i == j: continue
            ed = e_dist[i] - e_dist[j]
            td = t_dist[i] - t_dist[j]
            ra += (ed if abs(ed) > mu else ed + mu * td)
        scores.append(ra)
    return pd.Series(scores, index=norm_df.index)

def get_status(score, modal):
    if score >= 0.81:
        return labels[lang]['status']['sangat_layak'], modal * 0.60
    if score >= 0.61:
        return labels[lang]['status']['layak'], modal * 0.45
    if score >= 0.41:
        return labels[lang]['status']['cukup_layak'], modal * 0.30
    if score >= 0.21:
        return labels[lang]['status']['kurang_layak'], modal * 0.15
    return labels[lang]['status']['tidak_layak'], 0.0

def validate_csv_columns(df, expected):
    return all(col in df.columns for col in expected)

# === MAIN LOGIC ===
st.title(labels[lang]['title'])
df_usaha = None

# Input Manual
if input_method == "Manual":
    st.subheader(labels[lang]['manual'])
    n = st.number_input(labels[lang]['num_usaha'], 1, 20, 1)
    default = pd.DataFrame({
        labels[lang]['nama_usaha']: [f"Usaha {i+1}" for i in range(n)],
        **{c: [0.0]*n for c in labels[lang]['kriteria']}
    })
    editor = st.data_editor(default, use_container_width=True, num_rows="dynamic")
    if st.button(labels[lang]['save']):
        df_usaha = editor.copy()

# Input CSV
elif input_method == "Upload":
    st.subheader(labels[lang]['upload'])
    tmpl = pd.DataFrame({
        labels[lang]['nama_usaha']: [""],
        **{c: [0.0] for c in labels[lang]['kriteria']}
    })
    st.download_button(
        labels[lang]['download_template'],
        tmpl.to_csv(index=False).encode(),
        'template_input_usaha.csv',
        'text/csv'
    )
    up = st.file_uploader(labels[lang]['upload_prompt'], type=["csv"])
    if up:
        cand = pd.read_csv(up)
        expect = [labels[lang]['nama_usaha']] + labels[lang]['kriteria']
        if validate_csv_columns(cand, expect):
            df_usaha = cand.copy()
        else:
            st.error(labels[lang]['error_kolom'])

# Proses & Tampilkan
if df_usaha is not None:
    st.subheader(labels[lang]['data_usaha'])
    st.dataframe(df_usaha, use_container_width=True)

    df_kriter = df_usaha[labels[lang]['kriteria']].apply(pd.to_numeric, errors='coerce').fillna(0)
    w, norm = calculate_critic(df_kriter, cost_cols=["Modal Awal (Rp)", "Tingkat Risiko (1-5)"])
    st.subheader(labels[lang]['bobot'])
    st.write(w)

    # Hitung CODAS
    df_usaha["Skor CODAS"] = calculate_codas_full(norm, w).fillna(0)
    # Fill NaN sebelum ranking
    peringkat = df_usaha["Skor CODAS"].rank(ascending=False, method='min').fillna(0).astype(int)
    df_usaha["Peringkat"] = peringkat

    df_usaha["Status"], df_usaha["Rekomendasi Investasi (Rp)"] = zip(*[
        get_status(s, m) for s, m in zip(df_usaha["Skor CODAS"], df_kriter["Modal Awal (Rp)"])
    ])

    st.subheader(labels[lang]['hasil'])
    out = df_usaha[[
        labels[lang]['nama_usaha'],
        "Skor CODAS",
        "Peringkat",
        "Status",
        "Rekomendasi Investasi (Rp)"
    ]].sort_values("Peringkat").reset_index(drop=True)

    st.dataframe(
        out.style.format({
            "Skor CODAS": "{:.4f}",
            "Rekomendasi Investasi (Rp)": "Rp {:,.0f}"
        }),
        use_container_width=True
    )
    st.download_button(
        labels[lang]['download_hasil'],
        out.to_csv(index=False),
        "hasil_investasi.csv",
        "text/csv"
    )
