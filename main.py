# === SISTEM PENDUKUNG KEPUTUSAN INVESTASI BERBASIS DATABASE ===
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3

# === KONEKSI & INISIALISASI DATABASE ===
conn = sqlite3.connect('profil_usaha.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS profil_usaha (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nama TEXT NOT NULL,
        deskripsi TEXT,
        kategori TEXT
    )
''')
conn.commit()

# === LABEL BILINGUAL ===
if 'language' not in st.session_state:
    st.session_state.language = 'id'
lang = st.session_state.language

labels = {
    'id': {
        'title': "\U0001F4CA Sistem Pendukung Keputusan Investasi Usaha Mahasiswa",
        'manual': "\U0001F4DD Input Manual",
        'upload': "\U0001F4C1 Upload File",
        'profile': "\U0001F464 Profil Usaha",
        'num_usaha': "Jumlah Usaha",
        'save': "\U0001F4BE Simpan & Tampilkan Hasil",
        'download_template': "⬇️ Unduh Template Kosong (CSV)",
        'upload_prompt': "Unggah file CSV",
        'data_usaha': "\U0001F4C4 Data Usaha Mahasiswa",
        'bobot': "\U0001F4CC Bobot Kriteria (Metode CRITIC)",
        'hasil': "\U0001F4C8 Hasil Rekomendasi Investasi",
        'download_hasil': "\U0001F4BE Unduh Hasil",
        'change_lang': "\U0001F1EC\U0001F1E7 English",
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
        'profile': "\U0001F464 Business Profiles",
        'num_usaha': "Number of Businesses",
        'save': "\U0001F4BE Save & Show Result",
        'download_template': "⬇️ Download Blank Template (CSV)",
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

# Fungsi DB
def insert_profil(nama, deskripsi, kategori):
    cursor.execute("INSERT INTO profil_usaha (nama, deskripsi, kategori) VALUES (?, ?, ?)",
                   (nama, deskripsi, kategori))
    conn.commit()

def get_all_profiles():
    cursor.execute("SELECT * FROM profil_usaha")
    return cursor.fetchall()

def delete_profile(profile_id):
    cursor.execute("DELETE FROM profil_usaha WHERE id = ?", (profile_id,))
    conn.commit()

# Fungsi Analisis
standard_kriteria = ['ROI (%)', 'Modal Awal (Rp)', 'Pendapatan Rata-Rata 3 Bulan (Rp)',
                     'Aset (Rp)', 'Inovasi Produk (1-5)', 'Peluang Pasar (1-5)', 'Tingkat Risiko (1-5)']

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
