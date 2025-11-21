# ============================================================
# 🏍️ MOTORBIKE RECOMMENDATION DASHBOARD (Pandora Blue – Dark Mode)
# ============================================================
# Author: Hai Nguyen
# Version: v2 – Hybrid TF-IDF + Word2Vec
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 🧭 PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🏍️ Motorbike Recommendation Dashboard",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 🎨 PANDORA BLUE – DARK MODE (FULL LAYOUT SYNC + FIXED TOGGLE)
# ============================================================
st.markdown("""
<style>
/* === GLOBAL DARK PANDORA BLUE THEME === */
html, body, .stApp, .stAppViewContainer, .stAppViewBlockContainer, .main, .block-container {
    background-color: #0F172A !important;  /* Deep blue base */
    color: #E2E8F0 !important;
    font-family: "Segoe UI", sans-serif;
}

/* --- Sidebar --- */
aside[data-testid="stSidebar"] {
    background-color: #1E293B !important;  /* Slightly lighter navy */
    color: #E2E8F0 !important;
    font-weight: 500;
    border-right: 1px solid #334155 !important;
}
.stSidebar [role="radiogroup"] label {
    color: #E2E8F0 !important;
    font-weight: 500;
}

/* --- Sidebar toggle (mũi tên) --- */
[data-testid="stSidebarCollapseButton"] svg {
    color: #93C5FD !important;   /* Bright blue arrow */
}
[data-testid="stSidebarCollapseButton"]:hover svg {
    color: #60A5FA !important;
}

/* --- Titles --- */
h1, h2, h3, h4 {
    font-family: 'Segoe UI Semibold', sans-serif;
    color: #93C5FD !important;
}

/* --- Buttons --- */
div.stButton > button:first-child {
    background-color: #2563EB !important;
    color: #F8FAFC !important;
    border-radius: 10px;
    height: 42px;
    font-weight: bold;
    border: 1px solid #3B82F6 !important;
}
div.stButton > button:first-child:hover {
    background-color: #3B82F6 !important;
}

/* --- Slider (thanh trượt) --- */
[data-testid="stSlider"] > div > div {
    background: #334155 !important; /* Track */
}
[data-testid="stSlider"] div[role="slider"] {
    background: #60A5FA !important; /* Knob */
    border: 2px solid #F1F5F9 !important;
}

/* --- Text input --- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #1E293B !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
}

/* --- Metric values --- */
[data-testid="stMetricValue"] {
    color: #60A5FA !important;
    font-weight: 700;
}

/* --- Tables and DataFrames --- */
div[data-testid="stDataFrame"] .st-bd {
    color: #E2E8F0 !important;
    background-color: #1E293B !important;
}

/* --- Remove white header bar --- */
div[data-testid="stHeader"] {
    background: none !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* --- Remove or recolor the black top header bar --- */
div[data-testid="stHeader"] {
    background-color: #0F172A !important;   /* cùng màu nền chính */
    color: #E2E8F0 !important;
    border: none !important;
    height: 0rem !important;  /* hoặc 0 để ẩn luôn */
}
div[data-testid="stToolbar"] {
    background-color: #0F172A !important;
    border: none !important;
}
.viewerBadge_container__1QSob {
    background: transparent !important;
    color: transparent !important;
}
</style>
""", unsafe_allow_html=True)
# --- Nếu chỉ có file .npy thì tạo bản .index đúng chuẩn gensim ---
import os, shutil

npy_path = "model/tfidf_index.index.index.npy"
gensim_index_path = "model/tfidf_index.index"

if os.path.exists(npy_path) and not os.path.exists(gensim_index_path):
    shutil.move(npy_path, gensim_index_path)
    st.success("✅ Renamed tfidf_index.index.index.npy → tfidf_index.index")


# ============================================================
# 🔧 LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    import io

    file_path = "data/motorbike_clean.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    if not content:
        raise ValueError("❌ CSV file is empty or unreadable. Check encoding or upload again.")

    df = pd.read_csv(io.StringIO(content))
    print(f"✅ Loaded dataset successfully: {df.shape}")

    dictionary = corpora.Dictionary.load("model/dictionary.dict")
    tfidf_model = models.TfidfModel.load("model/tfidf_gensim.model")
    index = similarities.MatrixSimilarity.load("model/tfidf_index.index")
    texts = joblib.load("model/texts.pkl")
    model_w2v = Word2Vec.load("model/w2v_model.pkl")
    return df, dictionary, tfidf_model, index, texts, model_w2v
# 🧩 AUTO REBUILD TFIDF INDEX (for any number of parts)
# ============================================================

import os, zipfile

zip_path = "model/tfidf_index_index.zip"
target_file = "model/tfidf_index.index.index.npy"

# Nếu file .npy chưa tồn tại thì giải nén
if os.path.exists(zip_path) and not os.path.exists(target_file):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("model")
    print("✅ Extracted tfidf_index_index.zip → model/")
else:
    print("⚙️ tfidf_index.index.index.npy ready.")



df, dictionary, tfidf_model, index, texts, model_w2v = load_model()

# ============================================================
# 🔍 RECOMMENDER FUNCTION
# ============================================================
def get_vector(words, model):
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

def recommend_hybrid(query, df, dictionary, tfidf_model, index, model_w2v, texts, top_n=30, final_k=5):
    query_tokens = simple_preprocess(query)
    vec_tfidf = tfidf_model[dictionary.doc2bow(query_tokens)]
    sims_tfidf = index[vec_tfidf]
    top_idx = np.argsort(sims_tfidf)[::-1][:top_n]

    doc_vecs = np.array([get_vector(texts[i], model_w2v) for i in top_idx])
    query_vec = get_vector(query_tokens, model_w2v).reshape(1, -1)
    sims_w2v = cosine_similarity(query_vec, doc_vecs).flatten()
    sims_mix = 0.6 * sims_w2v + 0.4 * np.array(sims_tfidf[top_idx])

    boost = []
    query_lower = query.lower()
    for i in top_idx:
        title = str(df.iloc[i]["Tiêu đề"]).lower()
        brand = str(df.iloc[i]["Thương hiệu"]).lower()
        brand_match = 1.15 if any(b in query_lower for b in brand.split()) else 1.0
        model_match = 1.25 if any(m in title for m in query_lower.split()) else 1.0
        boost.append(brand_match * model_match)
    sims_final = sims_mix * np.array(boost)

    best_idx = np.argsort(sims_final)[::-1][:final_k]
    selected_idx = [top_idx[i] for i in best_idx]

    res = df.iloc[selected_idx][["Tiêu đề", "Giá", "Thương hiệu", "Loại xe"]].copy()
    res["similarity"] = np.round(sims_final[best_idx], 3)
    return res.reset_index(drop=True)

# ============================================================
# 🧭 SIDEBAR NAVIGATION
# ============================================================
menu = st.sidebar.radio(
    "🧭 Menu",
    [
        "🏁 Business Overview",
        "🔍 Recommendation",
        "📊 Model Analysis"
    ],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.caption("💡 Click the arrow **< / >** to show or hide the sidebar.")

# ============================================================
# 1️⃣ BUSINESS OVERVIEW
# ============================================================
if menu == "🏁 Business Overview":
    st.header("🎯 Business Problem & App Purpose")
    st.markdown("""
    This app helps users **find similar motorbikes** based on descriptions —  
    for example: *Yamaha Grande 125cc blue scooter, owner used*.

    **Objectives:**
    - Recommend similar bikes based on textual similarity.  
    - Combine TF-IDF (keyword relevance) and Word2Vec (semantic meaning).  
    - Provide interpretable similarity scores for each match.

    **Technology Stack:**
    - Python, Gensim, Scikit-learn, Pandas, Streamlit  
    - Visualization: Matplotlib, Seaborn  
    - Text Processing: underthesea, custom Vietnamese dictionaries  
    """)

# ============================================================
# 2️⃣ RECOMMENDATION
# ============================================================
elif menu == "🔍 Recommendation":
    st.header("🔍 Find Similar Motorbikes")
    st.markdown("----")
    query = st.text_input("Enter bike description:", "xe tay ga yamaha grande 125cc xanh")
    k = st.slider("Number of recommendations:", 3, 10, 5, step=1)

    if st.button("🚀 Recommend"):
        with st.spinner("Analyzing text and finding similar bikes..."):
            results = recommend_hybrid(query, df, dictionary, tfidf_model, index, model_w2v, texts, top_n=30, final_k=k)

        st.success(f"✅ Top {k} similar bikes for: **{query}**")
        st.dataframe(results, use_container_width=True)
        st.markdown("---")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=results, x="similarity", y="Tiêu đề", palette="crest", ax=ax)
        ax.set_title("Similarity Scores of Recommended Bikes", color="#93C5FD", weight="bold")
        st.pyplot(fig)

# ============================================================
# 3️⃣ MODEL ANALYSIS
# ============================================================
else:
    st.header("📊 Model Analysis & Insights")
    st.markdown("----")

    col1, col2 = st.columns(2)
    with col1:
        top_brands = df["Thương hiệu"].value_counts().head(10)
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        sns.barplot(y=top_brands.index, x=top_brands.values, palette="rocket", ax=ax1)
        ax1.set_title("Top 10 Popular Brands", color="#93C5FD", weight="bold")
        st.pyplot(fig1)
    with col2:
        top_types = df["Loại xe"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        sns.barplot(y=top_types.index, x=top_types.values, palette="magma", ax=ax2)
        ax2.set_title("Motorbike Type Distribution", color="#93C5FD", weight="bold")
        st.pyplot(fig2)

    df["mo_ta_len"] = df["content"].apply(lambda x: len(str(x).split()))
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.histplot(df["mo_ta_len"], bins=60, kde=True, color="#60A5FA", ax=ax3)
    ax3.set_title("Distribution of Description Length", color="#93C5FD", weight="bold")
    st.pyplot(fig3)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
# st.caption("📘 Hybrid Recommendation App – TF-IDF + Word2Vec | Designed by Hai Nguyen")
