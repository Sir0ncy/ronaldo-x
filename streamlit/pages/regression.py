import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_PATH = "regresi/model_regresi.pkl"
DATA_PATH = "regresi/data_regresi_twitter.csv"
TARGET = "retweet_count"
FITUR = ["favorite_count", "has_image", "word_count"]

st.set_page_config(page_title="Prediksi Retweet", layout="wide")
st.title("Prediksi Retweet Postingan X (Twitter)")
st.caption("Model regresi linear berganda, data dari tab Top.")

@st.cache_resource
def muat_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def muat_data():
    return pd.read_csv(DATA_PATH)

model = muat_model()
df = muat_data()

data = df[FITUR + [TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
y_aktual = data[TARGET]
y_pred = np.clip(model.predict(data[FITUR]), 0, None)

mae = mean_absolute_error(y_aktual, y_pred)
r2 = r2_score(y_aktual, y_pred)
mae_baseline = mean_absolute_error(y_aktual, np.full(len(y_aktual), y_aktual.median()))

st.header("Evaluasi Model")
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:,.1f}", help="Rata-rata selisih prediksi dan retweet sebenarnya")
c2.metric("R²", f"{r2:.3f}", help="Proporsi variasi retweet yang dijelaskan model")
c3.metric("MAE tebak median", f"{mae_baseline:,.1f}", help="Pembanding: jika selalu menebak median retweet")

st.info(
    "Catatan: R² tinggi sebagian besar didorong tweet viral (data dari tab Top), "
    "sehingga model hanya cocok untuk tweet populer, bukan semua tweet."
)

st.subheader("Grafik Aktual vs Prediksi")
skala_log = st.checkbox("Tampilkan skala log (pola tweet biasa lebih terlihat)", value=True)

fig, ax = plt.subplots(figsize=(6, 5))
x, y = (np.log1p(y_aktual), np.log1p(y_pred)) if skala_log else (y_aktual, y_pred)
ax.scatter(x, y, alpha=0.5, s=20, color="darkblue")
batas = [min(x.min(), y.min()), max(x.max(), y.max())]
ax.plot(batas, batas, color="red", linestyle="--", label="Prediksi sempurna")
ax.set_xlabel("Retweet aktual" + (" (log1p)" if skala_log else ""))
ax.set_ylabel("Retweet prediksi" + (" (log1p)" if skala_log else ""))
ax.legend()
st.pyplot(fig)
st.caption("Semakin dekat titik ke garis merah, semakin akurat prediksinya.")

st.header("Coba Prediksi")
with st.form("form_prediksi"):
    teks = st.text_area("Teks postingan", placeholder="Tulis isi postingan di sini...", max_chars=280)
    favorit = st.number_input("Perkiraan jumlah favorit (like)", min_value=0, value=1000, step=100)
    ada_gambar = st.checkbox("Postingan memuat gambar")
    kirim = st.form_submit_button("Hitung prediksi")

if kirim:
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
        st.caption(
            f"Perkiraan kesalahan rata-rata ±{mae:,.0f} retweet (MAE). "
            "Hasil menunjukkan hubungan statistik, bukan sebab-akibat."
        )