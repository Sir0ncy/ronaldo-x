import streamlit as st
import pandas as pd

# Load dataset
dataset_path = 'ronaldo_labeled.csv'
data = pd.read_csv(dataset_path)

st.title("Analisis Sentimen Ronaldo")

# 1. Jumlah dan persentase postingan setiap kategori
st.write("### Jumlah Sentimen")
sentiment_counts = data['sentimen'].value_counts()
col1, col2, col3 = st.columns(3)
col1.metric("Negative", sentiment_counts.get("Negative", 0))
col1.metric("Persentase" , f"{sentiment_counts.get('Negative', 0) / len(data) * 100:.2f}%")
col2.metric("Neutral", sentiment_counts.get("Neutral", 0))
col2.metric("", f"{sentiment_counts.get('Neutral', 0) / len(data) * 100:.2f}%")
col3.metric("Positive", sentiment_counts.get("Positive", 0))
col3.metric("", f"{sentiment_counts.get('Positive', 0) / len(data) * 100:.2f}%")

# 2. Grafik distribusi sentimen.
st.write("### Distribusi Sentimen")
st.bar_chart(sentiment_counts)

# 3. Tabel teks postingan beserta hasil klasifikasinya.
st.write("### Dataset")
st.dataframe(data)

# 20 random samples from the dataset
st.write("### 20 Data Random dari Dataset")
df_random = data.sample(n=20, random_state=44)
df_random = df_random.drop(columns=['confidence'])
st.dataframe(df_random)
