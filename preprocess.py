# ============================================================
# 📘 preprocess.py – Tiền xử lý văn bản cho 2 hướng TF-IDF & W2V
# ============================================================
import pandas as pd, re, unicodedata, os
from underthesea import word_tokenize

# --- Load dữ liệu ---
def load_data(path="data/data_motorbikes.xlsx"):
    df = pd.read_excel(path)
    df = df.dropna(subset=["Tiêu đề", "Mô tả chi tiết"])
    print(f"✅ Đọc thành công {len(df)} dòng từ {path}")
    return df

# --- Bỏ dấu ---
def remove_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return str(text)

# --- Load dictionary ---
def load_dict(path):
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return dict(line.split(":") for line in lines if ":" in line)

# ============================================================
# 1️⃣ FULL CLEAN – cho TF-IDF (chuẩn hóa mạnh)
# ============================================================
def clean_text_full(text, stop_words, teen_dict, eng_dict, wrong_dict):
    if pd.isnull(text): return ""
    text = str(text).lower()

    # ✅ Thay teen code, từ sai, từ English
    for w, c in {**wrong_dict, **teen_dict, **eng_dict}.items():
        text = re.sub(rf"\b{re.escape(w)}\b", c, text)

    # ✅ Tách từ giữ cụm nghĩa
    text = word_tokenize(text, format="text")

    # ✅ Bỏ dấu, ký tự đặc biệt
    text = remove_accents(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # ✅ Loại stopword
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

# ============================================================
# 2️⃣ LIGHT CLEAN – cho Gensim + Word2Vec (giữ nghĩa)
# ============================================================
def clean_text_light(text):
    if pd.isnull(text): return ""
    text = str(text).lower()
    text = word_tokenize(text, format="text")      # giữ dấu
    text = re.sub(r"[^a-zA-Z0-9áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệ"
                  r"íìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữự"
                  r"ýỳỷỹỵđ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================================
# 3️⃣ Pipeline tiền xử lý
# ============================================================
def preprocess_data(df, mode="light"):
    if mode == "full":
        stop_words = [w.strip() for w in open("files/vietnamese-stopwords.txt", "r", encoding="utf-8")] \
            if os.path.exists("files/vietnamese-stopwords.txt") else []
        teen_dict  = load_dict("files/teencode.txt")
        eng_dict   = load_dict("files/english-vnmese.txt")
        wrong_dict = load_dict("files/wrong-word.txt")
        func = lambda x: clean_text_full(x, stop_words, teen_dict, eng_dict, wrong_dict)
    else:
        func = clean_text_light

    text_cols = ["Tiêu đề", "Thương hiệu", "Dòng xe", "Loại xe", "Dung tích xe", "Mô tả chi tiết"]
    for col in text_cols:
        df[col] = df[col].apply(func)

    df["content"] = df[text_cols].agg(" ".join, axis=1)
    print(f"✅ Hoàn tất tiền xử lý ({mode.upper()} mode).")
    return df

# ============================================================
# 4️⃣ Chạy thử
# ============================================================
if __name__ == "__main__":
    df = load_data()
    # ⚙️ Thay 'full' bằng 'light' tùy mục tiêu
    df = preprocess_data(df, mode="light")
    df.to_csv("data/motorbike_clean.csv", index=False)
    print("💾 Đã lưu data/motorbike_clean.csv")
