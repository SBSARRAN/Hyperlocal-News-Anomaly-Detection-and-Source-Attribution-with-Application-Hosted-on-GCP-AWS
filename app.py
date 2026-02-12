import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hyperlocal News Anomaly Detection", layout="wide")

st.title("📰 Hyperlocal News Anomaly Detection System")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("final_news_output.csv")

    # --- Fix Boolean Columns ---
    bool_cols = ["Source_Mismatch", "Final_Anomaly"]

    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .map({"True": True, "False": False})
            )
            df[col] = df[col].fillna(False)

    return df


df = load_data()

# ===============================
# SIDEBAR FILTERS
# ===============================
st.sidebar.header("🔍 Filters")

anomaly_filter = st.sidebar.selectbox(
    "Select Anomaly Type",
    ["All", "Normal", "Anomaly"]
)

source_filter = st.sidebar.selectbox(
    "Source Mismatch",
    ["All", True, False]
)

final_filter = st.sidebar.selectbox(
    "Final Anomaly",
    ["All", True, False]
)

filtered_df = df.copy()

# Filter anomaly_label
if anomaly_filter != "All":
    filtered_df = filtered_df[
        filtered_df["anomaly_label"] == anomaly_filter
    ]

# Filter Source_Mismatch
if source_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Source_Mismatch"] == source_filter
    ]

# Filter Final_Anomaly
if final_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Final_Anomaly"] == final_filter
    ]

# ===============================
# KPI METRICS
# ===============================
st.subheader("📊 System Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Articles", len(df))
col2.metric("Detected Anomalies", df["Final_Anomaly"].sum())
col3.metric("Source Mismatch Cases", df["Source_Mismatch"].sum())

# ===============================
# ANOMALY DISTRIBUTION CHART
# ===============================
st.subheader("📈 Anomaly Distribution")

fig = plt.figure()
df["Final_Anomaly"].value_counts().plot(kind="bar")
plt.xticks(rotation=0)
plt.xlabel("Final Anomaly")
plt.ylabel("Count")
st.pyplot(fig)

# ===============================
# FILTERED DATA TABLE
# ===============================
st.subheader("📑 Filtered Articles")

st.dataframe(filtered_df, use_container_width=True)

# ===============================
# ARTICLE DETAIL VIEW
# ===============================
st.subheader("📝 Article Details")

if len(filtered_df) > 0:

    selected_index = st.selectbox(
        "Select Article Index",
        filtered_df.index
    )

    article = filtered_df.loc[selected_index]

    st.markdown(f"### {article['Heading']}")
    st.markdown(f"**Date:** {article['Date']}")
    st.markdown(f"**News Type:** {article['NewsType']}")
    st.markdown(f"**Header Location:** {article['Header_Location']}")
    st.markdown(f"**NER Locations:** {article['NER_Location']}")
    st.markdown(f"**Predicted Location:** {article['Predicted_Location']}")
    st.markdown(f"**Sentiment Label:** {article['sentiment_label']}")
    st.markdown(f"**Anomaly Label:** {article['anomaly_label']}")
    st.markdown(f"**Source Mismatch:** {article['Source_Mismatch']}")
    st.markdown(f"**Discrepancy Score:** {round(article['Discrepancy_Score'], 3)}")
    st.markdown(f"**Final Anomaly:** {article['Final_Anomaly']}")

    st.text_area("Article Text", article["Article"], height=300)

else:
    st.warning("No articles match selected filters.")

