import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Resolve paths relative to this script so the app runs from any working directory.
# This file lives in <project>/streamlit/pages/, so the project root is two levels up.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

MODEL_PATH = os.path.join(PROJECT_DIR, "regresi", "model_regresi.pkl")
DATA_PATH = os.path.join(PROJECT_DIR, "regresi", "data_regresi_twitter.csv")
TARGET = "retweet_count"
FITUR = ["favorite_count", "has_image", "word_count"]
VIF_COLS = ["posting_hour", "text_length", "favorite_count",
            "reply_count", "quote_count", "has_image", "sentiment_code"]
THRESHOLD = 0.80
TEST_SIZE = 0.2  
RANDOM_STATE = 42     

st.set_page_config(page_title="Prediksi Retweet", layout="wide")
st.title("Prediksi Retweet Postingan X (Twitter)")
st.caption("Model regresi linear berganda, data dari tweet 'Ronaldo' tab Top.")
 
@st.cache_resource
def muat_model():
    return joblib.load(MODEL_PATH)
 
 
@st.cache_data
def muat_data():
    return pd.read_csv(DATA_PATH)
 
 
model = muat_model()
dataset = muat_data()
 
with st.container(border=True):
    st.write(f"Jumlah baris: **{dataset.shape[0]}**, jumlah kolom: **{dataset.shape[1]}**")
    st.dataframe(dataset, use_container_width=True, height=300)
    
    st.write("---\nKeterangan:")
    cola, colb = st.columns(2)
    with cola:
        with st.container(border=True):
            st.markdown("1. Kolom `favorite_count`, `retweet_count`, `reply_count`, `quote_count` diambil langsung dari dataset mentah.")
    with colb:
        with st.container(border=True):
            st.markdown("2. Kolom `text_length`, `word_count`, `posting_hour`, `has_image`, dan `sentiment_code` dari ekstraksi dataset.")
    
st.header("# Cek Multikolinearitas")
with st.container(border=True):
    st.write(
        "**Multikolinearitas** terjadi saat sesama variabel X saling berkorelasi kuat, "
        "sehingga koefisien regresi tidak stabil. Dicek dengan korelasi Pearson dan VIF."
    )

    df_fitur = dataset.select_dtypes(include="number")
    
    # Heatmap
    st.subheader("1. Heatmap Korelasi")
    matriks_korelasi = df_fitur.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matriks_korelasi, annot=True, cmap="Oranges", fmt=".2f",
                linewidths=0.5, square=True, ax=ax)
    ax.set_title("Heatmap Korelasi Antar Variabel Numerik Tweet", fontsize=14, pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    
    # VIF
    st.subheader("2. VIF (Variance Inflation Factor)")
    vif_cols = [c for c in VIF_COLS if c in dataset.columns]
    X_vif = dataset[vif_cols].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    X_vif = sm.add_constant(X_vif)
    vif = pd.DataFrame({
        "Variabel": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
    })
    vif["Status"] = np.where(vif["Variabel"] == "const", "abaikan",
                    np.where(vif["VIF"] > 10, "serius (>10)",
                    np.where(vif["VIF"] >= 5, "waspada (5-10)", "aman (<5)")))
    st.dataframe(vif, use_container_width=True, hide_index=True)
    st.caption("VIF < 5 aman, 5-10 waspada, > 10 multikolinearitas serius. Baris const diabaikan.")

# Choose features and target
st.header("# Variabel yang Dipilih")
with st.container(border=True):
    keterangan = pd.DataFrame([
        ["retweet_count", "Y", "Variabel yang diprediksi"],
        ["favorite_count", "X", "VIF aman, hubungan kuat dan linear dengan Y"],
        ["has_image", "X", "Variabel biner (1 = ada gambar)"],
        ["word_count", "X", "Jumlah kata teks postingan"],
    ], columns=["Variabel", "Peran", "Alasan"])
    st.dataframe(keterangan, use_container_width=True, hide_index=True)
 
# Detail training spesification and training regression model
st.header("# Spesifikasi Training")
with st.container(border=True):
    data = dataset[FITUR + [TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        data[FITUR], data[TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    spesifikasi = pd.DataFrame({
        "Item": ["Algoritma", "Variabel Y", "Variabel X", "Jumlah data (bersih)",
                "Data latih", "Data uji", "Test size", "Random state", "Skala data"],
        "Nilai": [type(model).__name__, TARGET, ", ".join(FITUR), len(data),
                len(X_train), len(X_test), TEST_SIZE, RANDOM_STATE,
                "Asli (tanpa transformasi log)"],
    })
    st.dataframe(spesifikasi, use_container_width=True, hide_index=True)
 
# Results
st.header("# Hasil Model")
with st.container(border=True):
    y_pred = np.clip(model.predict(X_test), 0, None)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae_base = mean_absolute_error(y_test, np.full(len(y_test), y_train.median()))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (data uji)", f"{mae:,.1f}", help="Rata-rata selisih prediksi dan retweet sebenarnya")
    c2.metric("R² (data uji)", f"{r2:.3f}", help="Proporsi variasi retweet yang dijelaskan model")
    c3.metric("MAE tebak median", f"{mae_base:,.1f}", help="Pembanding: jika selalu menebak median")
    
    st.info(
        "R² tinggi sebagian besar didorong tweet viral (data tab Top). "
        "Model hanya cocok untuk tweet populer, tidak untuk semua tweet."
    )
    st.markdown("#### Pengaruh Variabel X terhadap Retweet")
    pengaruh = pd.DataFrame({"Variabel": FITUR, "Koefisien": model.coef_})
    pengaruh["Arti"] = [
        f"Tiap +1 {v} berhubungan dengan perubahan retweet sekitar {k:+.3f}"
        for v, k in zip(pengaruh["Variabel"], pengaruh["Koefisien"])
    ]
    st.dataframe(pengaruh, use_container_width=True, hide_index=True)
    st.write(f"Intersep: **{float(model.intercept_):.2f}**")
    st.caption("Koefisien menunjukkan hubungan statistik, bukan sebab-akibat.")
 
# Check actual vs predicted plot
st.subheader("# Grafik Aktual vs Prediksi")
with st.container(border=True):
    # Using log scale to make the pattern more visible, especially for non-viral tweets
    skala_log = st.checkbox("Skala log (pola tweet biasa lebih terlihat)", value=True)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    xa, ya = (np.log1p(y_test), np.log1p(y_pred)) if skala_log else (y_test, y_pred)
    ax2.scatter(xa, ya, alpha=0.5, s=20, color="darkblue")
    batas = [min(np.min(xa), np.min(ya)), max(np.max(xa), np.max(ya))]
    ax2.plot(batas, batas, "r--", label="Prediksi sempurna")
    ax2.set_xlabel("Retweet aktual" + (" (log1p)" if skala_log else ""))
    ax2.set_ylabel("Retweet prediksi" + (" (log1p)" if skala_log else ""))
    ax2.legend()
    st.pyplot(fig2)
    st.caption("Semakin dekat titik ke garis merah, semakin akurat prediksinya.")
 
# ------------------------------------------------------------- 6. FORM
st.header("# Coba Prediksi")
with st.form("form_prediksi"):
    teks = st.text_area("Teks postingan", max_chars=280,
                        placeholder="Tulis isi postingan di sini...")
    favorit = st.number_input("Perkiraan jumlah favorit (like)", min_value=0, value=1000, step=100)
    ada_gambar = st.checkbox("Postingan memuat gambar")
    kirim = st.form_submit_button("Hitung prediksi")
 
if kirim:
    with st.container(border=True):
        if not teks.strip():
            st.warning("Teks postingan tidak boleh kosong.")
        else:
            jumlah_huruf = len(teks)
            jumlah_kata = len(teks.split())
            a, b = st.columns(2)
            a.metric("Jumlah huruf (karakter)", jumlah_huruf)
            b.metric("Jumlah kata", jumlah_kata)
    
            nilai = {
                "favorite_count": favorit,
                "has_image": int(ada_gambar),
                "word_count": jumlah_kata,
                "text_length": jumlah_huruf,
            }
            X_baru = pd.DataFrame([[nilai[f] for f in FITUR]], columns=FITUR)
            hasil = max(float(model.predict(X_baru)[0]), 0)
    
            st.success(f"Prediksi retweet: sekitar **{hasil:,.0f}**")
            st.caption(f"Perkiraan kesalahan rata-rata ±{mae:,.0f} retweet (MAE).")
    