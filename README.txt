📘 MOTORBIKE RECOMMENDATION PROJECT
===================================

1️⃣ Mục tiêu:
   - Gợi ý xe máy tương tự dựa trên mô tả, thương hiệu, dòng xe.
   - So sánh 3 mô hình: TF-IDF (Sklearn), TF-IDF (Gensim), Word2Vec.

2️⃣ Cấu trúc project:
   app.py                → Giao diện Streamlit (hiển thị kết quả)
   preprocess.py         → Làm sạch & chuẩn hóa dữ liệu
   recommender.py        → Huấn luyện mô hình gợi ý
   data/data_motorbikes.xlsx → Dữ liệu gốc

3️⃣ Quy trình huấn luyện:
   python preprocess.py   → Tạo motorbike_clean.csv
   python recommender.py  → Huấn luyện TF-IDF, Gensim, W2V

4️⃣ Cách chạy GUI (sẽ thêm sau):
   streamlit run app.py

5️⃣ Output mô hình:
   model/tfidf_vectorizer.pkl
   model/w2v_model.pkl
   model/tfidf_matrix.npy
