import os

import pandas as pd
import streamlit as st

# Resolve paths relative to this script so the app runs from any working directory.
# This file lives in <project>/streamlit/pages/, so the project root is two levels up.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

st.set_page_config(
    page_title="Preprocessing",
    layout="wide",
)

# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cleaned_data():
    """Load the cleaned CSV produced by the preprocessing pipeline."""
    path = os.path.join(PROJECT_DIR, "clean", "ronaldo_clean.csv")
    if not os.path.exists(path):
        path = os.path.join(PROJECT_DIR, "ronaldo-top.csv")
    return pd.read_csv(path, dtype={"id_str": str})


@st.cache_data(show_spinner=False)
def load_report():
    """Load the preprocessing report as plain text."""
    path = os.path.join(PROJECT_DIR, "clean", "preprocessing_report.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    return "Preprocessing report not found."


@st.cache_data(show_spinner=False)
def parse_report(text: str):
    """Parse the plain text report into a structured dictionary."""
    data = {
        "counts": {},
        "missing": [],
        "relevance": {},
        "validation": [],
    }
    if not text or "Preprocessing report not found" in text:
        return data

    lines = text.splitlines()
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "Initial row count:" in line:
            data["counts"]["initial"] = line.split(":")[-1].strip()
            continue
        if "Duplicates removed" in line:
            key = line.split(":")[0].strip()
            data["counts"][key] = line.split(":")[-1].strip()
            continue
        if "Final row count" in line:
            data["counts"]["final"] = line.split(":")[-1].strip()
            continue

        if "Missing values per column" in line:
            current_section = "missing"
            continue
        if "Relevance results:" in line:
            current_section = "relevance"
            continue
        if "Validation:" in line:
            current_section = "validation"
            continue

        if current_section == "missing" and ":" in line:
            col, val = line.split(":", 1)
            data["missing"].append({"column": col.strip(), "count": val.strip()})
        elif current_section == "relevance" and ":" in line:
            label, val = line.split(":", 1)
            data["relevance"][label.strip()] = val.strip()
        elif current_section == "validation" and line.startswith("["):
            data["validation"].append(line)
        elif "Invalid/missing favorite_count" in line:
            data["counts"]["invalid_likes"] = line.split(":")[-1].strip()

    return data


df = load_cleaned_data()
report_text = load_report()
parsed_report = parse_report(report_text)

# Header
st.title("Ronaldo Tweet Preprocessing Insight")

# Engagement metrics
with st.container(border=True):
    st.subheader("Engagement metrics")
    with st.container(horizontal=True):
        if "favorite_count" in df.columns and df["favorite_count"].notna().any():
            st.metric(
                "Average likes",
                f"{df['favorite_count'].mean():,.0f}",
                border=True,
            )
        if "retweet_count" in df.columns:
            st.metric(
                "Average retweets",
                f"{df['retweet_count'].mean():,.0f}",
                border=True,
            )
        if "reply_count" in df.columns:
            st.metric(
                "Average replies",
                f"{df['reply_count'].mean():,.0f}",
                border=True,
            )

# Summary metrics and preprocessing report
with st.container(border=True):
    st.subheader("Preprocessing report")

    # 1. Key counts as metrics
    with st.container(horizontal=True):
        # Use parsed data for the report metrics, fallback to df for basics
        initial = parsed_report["counts"].get("initial", "N/A")
        final = parsed_report["counts"].get("final", len(df))
        st.metric("Initial rows", initial, border=True)
        st.metric("Final rows", final, border=True)
        duplicates = parsed_report["counts"].get("Duplicates removed by id_str", "0")
        st.metric("ID Duplicates", duplicates, border=True)

    # 2. Detailed breakdowns in tabs
    tab1, tab2, tab3 = st.tabs(["Missing Values", "Relevance", "Validation"])

    with tab1:
        if parsed_report["missing"]:
            missing_df = pd.DataFrame(parsed_report["missing"])
            missing_df["count"] = pd.to_numeric(missing_df["count"], errors="coerce")
            st.dataframe(missing_df, hide_index=True, alt="Table of missing values per column")
        else:
            st.info("No missing value data available in report.")

    with tab2:
        if parsed_report["relevance"]:
            rel_data = parsed_report["relevance"]
            rel_df = pd.DataFrame(
                [{"label": k, "count": int(v)} for k, v in rel_data.items()]
            )
            st.bar_chart(rel_df, x="label", y="count", alt="Bar chart of relevance results")
            st.dataframe(rel_df, hide_index=True, alt="Table of relevance results")
        else:
            st.info("No relevance data available in report.")

    with tab3:
        if parsed_report["validation"]:
            for item in parsed_report["validation"]:
                if "[PASS]" in item:
                    st.success(item.replace("[PASS]", "✅"))
                elif "[FAIL]" in item:
                    st.error(item.replace("[FAIL]", "❌"))
                else:
                    st.info(item)
        else:
            st.info("No validation data available in report.")

# Sample dataset
with st.container(border=True):
    st.subheader("Sample dataset")
    st.dataframe(
        df,
        hide_index=True,
        alt="The cleaned dataset",
    )

# Raw text vs clean text
with st.container(border=True):
    st.subheader("Raw text vs clean text")
    if "full_text" in df.columns and "text_clean" in df.columns:
        comparison = df[["full_text", "text_clean"]].copy()
        comparison.columns = ["Raw text", "Clean text"]
        st.dataframe(
            comparison,
            hide_index=True,
            alt="Raw tweet text alongside its cleaned version",
        )

# Column anomaly check
with st.container(border=True):
    st.subheader("Column anomaly check")
    rows = []
    for column in df.columns:
        series = df[column]
        nulls = int(series.isna().sum())
        if series.dtype == object:
            empties = int((series.astype(str).str.strip() == "").sum())
        else:
            empties = 0
        rows.append(
            {
                "Column": column,
                "Rows": len(series),
                "Null": nulls,
                "Empty": empties,
                "Null rate": round(nulls / len(series) * 100, 1),
                "Dtype": str(series.dtype),
            }
        )
    checks_df = pd.DataFrame(rows)
    st.dataframe(
        checks_df,
        hide_index=True,
        column_config={
            "Null rate": st.column_config.NumberColumn("Null rate", format="%.1f"),
        },
        alt="Per column anomaly summary with null and empty counts",
    )