# Ronaldo-X

Ronaldo-X adalah proyek analisis data tweets dari X.com yang berkaitan dengan Cristiano Ronaldo. Proyek ini mencakup data cleaning, preprocessing, analisis sentimen, clustering, dan regresi yang diimplementasikan menggunakan Python dan Streamlit.

## Fitur Utama

- **Data Cleaning & Preprocessing:** Membersihkan dan mempersiapkan data untuk analisis.
- **Analisis Sentimen:** Mengklasifikasikan teks menjadi sentimen positif, netral, dan negatif.
- **Clustering:** Mengelompokkan data untuk mengidentifikasi pola.
- **Regresi:** Pemodelan prediktif dengan pembagian data latih dan uji 70:30 serta analisis multikolinearitas.
- **Dashboard:** Menampilkan hasil analisis melalui aplikasi Streamlit.

## Struktur Proyek

```text
ronaldo-x/
├── .devcontainer/
├── clean/
├── clustering/
├── regresi/
├── ronaldo-top/
├── sentimen/
├── streamlit/
├── cleaning.py
├── preprocessing.py
├── preprocessing.md
├── requirements.txt
├── ronaldo.csv
└── ronaldo-top.csv
```

## Dataset

| File | Deskripsi |
|---|---|
| `ronaldo_clean.csv` | Dataset hasil cleaning. |
| `ronaldo_text_clean.csv` | Dataset teks yang telah dibersihkan. |
| `ronaldo_labeled.csv` | Dataset berlabel untuk analisis sentimen. |
| `ronaldo_manual_review.csv` | Dataset untuk peninjauan label manual. |
| `ronaldo_clustering.csv` | Dataset untuk clustering. |
| `data_regresi_twitter.csv` | Dataset untuk regresi. |

## Instalasi

Clone repository dan instal dependensi:

```bash
git clone https://github.com/SiR0ncy/ronaldo-x.git
cd ronaldo-x
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
streamlit run streamlit/app.py
```

Sesuaikan nama file jika file utama dashboard berbeda.

## Kontributor

- [@ahmadghani27](https://github.com/ahmadghani27)
- [@Sir0ncy](https://github.com/SiR0ncy)
- [@tiaaraaayy](https://github.com/tiaaraaayy)
- [@putmul](https://github.com/putmul)
