import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from wordcloud import WordCloud

# Resolve paths relative to this script so the app runs from any working directory.
# This file lives in <project>/streamlit/pages/, so the project root is two levels up.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

st.set_page_config(page_title="Ronaldo Tweet Clustering", layout="wide")
st.title("Ronaldo Tweet Clustering")

nltk.download("stopwords", quiet=True)

ADDITIONAL = {"ronaldo", "cristiano", "cr7", "https", "http", "amp", "rt", "www", "com", "via", "like", "time", "even", "football"}
FINAL_STOPWORDS = set(stopwords.words("english")).union(ADDITIONAL)
CLUSTER_THEMES = {0: "Fan Debate on Portugal and GOAT Status", 1: "Farewell to Messi", 2: "Jorge Jesus Controversy"}
COLORS = ["blue", "orange", "green"]

def load_data():
    data = pd.read_csv(os.path.join(PROJECT_DIR, "clustering", "ronaldo_clustering.csv"))
    data["text_clean"] = data["text_clean"].fillna("").astype(str)
    data["cluster_theme"] = data["cluster_theme"].fillna("").astype(str)
    data = data[data["text_clean"].str.strip().ne("") & data["cluster"].notna()].copy()
    data["cluster"] = data["cluster"].astype(int)
    return data.reset_index(drop=True)

def get_cluster_theme(cluster_data, cluster_id):
    if cluster_id in CLUSTER_THEMES:
        return CLUSTER_THEMES[cluster_id]
    themes = cluster_data["cluster_theme"].str.strip()
    themes = themes[themes != ""]
    if not themes.empty:
        return themes.value_counts().index[0]
    return f"Cluster {cluster_id}"

def main():
    data = load_data()
    cluster_values = sorted(data["cluster"].unique())

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words=list(FINAL_STOPWORDS), max_df=0.85, min_df=2, ngram_range=(1, 2), sublinear_tf=True)
    X = vectorizer.fit_transform(data["text_clean"])
    terms = vectorizer.get_feature_names_out()

    # Menghitung centroid setiap cluster
    centroids = np.vstack([np.asarray(X[data["cluster"].values == cluster_id].mean(axis=0)).ravel() for cluster_id in cluster_values])

    # PCA untuk visualisasi
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())
    centroid_coords = pca.transform(centroids)

    # Ringkasan dataset
    st.header("Ringkasan Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Postingan", f"{len(data):,}")
    col2.metric("Jumlah Cluster", len(cluster_values))
    col3.metric("Jumlah Fitur TF-IDF", f"{X.shape[1]:,}")
    st.divider()

    # Hasil setiap cluster
    st.header("Hasil Setiap Cluster")

    for i, cluster_id in enumerate(cluster_values):
        cluster_data = data[data["cluster"] == cluster_id]
        theme = get_cluster_theme(cluster_data, cluster_id)
        st.subheader(f"Cluster {cluster_id}: {theme}")

        # WordCloud
        cluster_text = " ".join(cluster_data["text_clean"])
        wc = WordCloud(stopwords=FINAL_STOPWORDS, width=1000, height=400, background_color="white", collocations=False).generate(cluster_text)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

        # Jumlah postingan
        persen = len(cluster_data) / len(data) * 100
        st.metric("Jumlah Postingan", f"{len(cluster_data):,} ({persen:.1f}%)")

        # Lima kata dominan
        st.markdown("#### Lima Kata Dominan")
        top_indices = centroids[i].argsort()[::-1][:5]
        top_words = [terms[index] for index in top_indices]
        st.write(", ".join(top_words))

        # Tiga contoh postingan
        st.markdown("#### Tiga Contoh Postingan")
        examples = cluster_data["text_clean"].sample(n=min(3, len(cluster_data)), random_state=42)
        for number, post in enumerate(examples, start=1):
            st.markdown(f"**{number}.** {post}")

        st.divider()

    # Visualisasi keseluruhan cluster
    st.header("Visualisasi Keseluruhan Cluster")
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, cluster_id in enumerate(cluster_values):
        mask = (data["cluster"] == cluster_id).values
        theme = get_cluster_theme(data[data["cluster"] == cluster_id], cluster_id)
        ax.scatter(coords[mask, 0], coords[mask, 1], alpha=0.6, color=COLORS[i % len(COLORS)], label=f"Cluster {cluster_id}: {theme}")

    ax.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker="X", s=250, c="black", edgecolors="white", linewidths=1.5, label="Centroid")
    ax.set_title("K-Means Clustering Visualization with Centroids")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(title="Cluster Themes", loc="upper right", fontsize=8, title_fontsize=9, frameon=True)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

if __name__ == "__main__":
    main()
