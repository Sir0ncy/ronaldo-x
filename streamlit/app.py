import os

import pandas as pd
import streamlit as st

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

st.set_page_config(
    page_title="Ronaldo Tweet Dashboard",
    page_icon=":material/dashboard:",
    layout="wide",
)

st.title("Ronaldo Tweet Analysis Dashboard")

# Quick overview of the cleaned dataset
@st.cache_data(show_spinner=False)
def load_cleaned_data():
    """Load the cleaned CSV produced by the preprocessing pipeline."""
    path = os.path.join(PROJECT_DIR, "clean", "ronaldo_clean.csv")
    if not os.path.exists(path):
        path = os.path.join(PROJECT_DIR, "ronaldo-top.csv")
    return pd.read_csv(path, dtype={"id_str": str})


df = load_cleaned_data()

with st.container(horizontal=True):
    st.metric("Total rows", len(df), border=True)
    st.metric("Total columns", len(df.columns), border=True)

cola, colb = st.columns(2)
with cola:
    st.image("https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA2eavev.img?w=768&h=474&m=6&x=95&y=133&s=825&d=249", width='stretch')
with colb:
    st.image("https://tse2.mm.bing.net/th/id/OIP.pAGKsZhSw2M7rVwx36E0ygAAAA?r=0&rs=1&pid=ImgDetMain&o=7&rm=3", width='stretch')