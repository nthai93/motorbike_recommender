# ============================================================
# 📗 recommender.py – Gensim TF-IDF + Word2Vec Hybrid Model
# ============================================================
import pandas as pd, numpy as np, os, joblib
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity

def load_clean_data(path="data/motorbike_clean.csv"):
    df = pd.read_csv(path)
    print(f"✅ Dữ liệu đã nạp: {df.shape}")
    return df

def build_gensim_tfidf(df, save_dir="model"):
    texts = [simple_preprocess(str(doc)) for doc in df["content"]]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]
    tfidf_model = models.TfidfModel(corpus)
    index = similarities.MatrixSimilarity(tfidf_model[corpus], num_features=len(dictionary))
    os.makedirs(save_dir, exist_ok=True)
    dictionary.save(os.path.join(save_dir, "dictionary.dict"))
    tfidf_model.save(os.path.join(save_dir, "tfidf_gensim.model"))
    index.save(os.path.join(save_dir, "tfidf_index.index"))
    joblib.dump(texts, os.path.join(save_dir, "texts.pkl"))
    print("💾 Đã lưu mô hình Gensim TF-IDF.")
    return dictionary, tfidf_model, index, texts

def build_w2v(texts, save_dir="model"):
    model_w2v = Word2Vec(sentences=texts, vector_size=150, window=5, min_count=2, sg=1, workers=4)
    model_w2v.save(os.path.join(save_dir, "w2v_model.pkl"))
    print("💾 Đã lưu mô hình Word2Vec.")
    return model_w2v

def get_vector(words, model):
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

def recommend_hybrid(query, df, dictionary, tfidf_model, index, model_w2v, texts, top_n=30, final_k=5):
    """
    Gợi ý xe tương tự (Hybrid: Gensim TF-IDF + Word2Vec)
    - TF-IDF lọc sơ bộ top_n
    - Word2Vec tính ngữ nghĩa
    - Boost theo thương hiệu và dòng xe
    """
    query_tokens = simple_preprocess(query)
    vec_tfidf = tfidf_model[dictionary.doc2bow(query_tokens)]
    sims_tfidf = index[vec_tfidf]
    top_idx = np.argsort(sims_tfidf)[::-1][:top_n]

    # --- Vector hóa Word2Vec ---
    doc_vecs = np.array([get_vector(texts[i], model_w2v) for i in top_idx])
    query_vec = get_vector(query_tokens, model_w2v).reshape(1, -1)
    sims_w2v = cosine_similarity(query_vec, doc_vecs).flatten()

    # --- Kết hợp TF-IDF + Word2Vec ---
    sims_mix = 0.6 * sims_w2v + 0.4 * np.array(sims_tfidf[top_idx])

    # --- Tăng trọng số nếu trùng thương hiệu hoặc dòng xe ---
    boost = []
    query_lower = query.lower()
    for i in top_idx:
        title = str(df.iloc[i]["Tiêu đề"]).lower()
        brand = str(df.iloc[i]["Thương hiệu"]).lower()
        # Nếu query chứa thương hiệu -> ưu tiên khớp brand
        brand_match = 1.15 if any(b in query_lower for b in brand.split()) else 1.0
        # Nếu query chứa tên model
        model_match = 1.25 if any(m in title for m in query_lower.split()) else 1.0
        boost.append(brand_match * model_match)

    sims_final = sims_mix * np.array(boost)

    # --- Lấy top K kết quả ---
    best_idx = np.argsort(sims_final)[::-1][:final_k]
    selected_idx = [top_idx[i] for i in best_idx]

    res = df.iloc[selected_idx][["Tiêu đề", "Giá", "Thương hiệu", "Loại xe"]].copy()
    res["similarity"] = np.round(sims_final[best_idx], 3)
    return res.reset_index(drop=True)


if __name__ == "__main__":
    df = load_clean_data()
    dictionary, tfidf_model, index, texts = build_gensim_tfidf(df)
    model_w2v = build_w2v(texts)
    query = "xe tay ga yamaha grande 125cc xanh"
    print(f"🔍 Gợi ý cho: {query}")
    print(recommend_hybrid(query, df, dictionary, tfidf_model, index, model_w2v, texts))
