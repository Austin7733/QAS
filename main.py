import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st
import nltk

# Download NLTK tokenizer
nltk.download("punkt")

# Model dan pipeline
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # Model lebih cepat
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Fungsi untuk memproses dan membagi teks
def preprocess_and_split_text(df):
    df["abstract"] = df["abstract"].str.replace('\n', ' ').str.replace('\r', '').str.strip()
    df["chunks"] = df["abstract"].apply(lambda x: nltk.sent_tokenize(x))
    return df

# Fungsi untuk membuat embeddings dan index FAISS
def create_faiss_index(df_chunks):
    all_chunks = []
    for _, row in df_chunks.iterrows():
        for chunk in row["chunks"]:
            all_chunks.append({"title": row["title"], "chunk": chunk})

    df_chunks = pd.DataFrame(all_chunks)
    embeddings = np.array([embedding_model.encode(chunk) for chunk in df_chunks["chunk"]])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return df_chunks, index

# Streamlit interface
st.title("Apa yang ingin anda ketahui tentang Machine Learning?")

uploaded_file = st.file_uploader("Silahkan upload file csv!", type=["csv"], accept_multiple_files=False)
if uploaded_file is not None:
    # Proses file CSV yang diupload
    df = pd.read_csv(uploaded_file)
    df_chunks = preprocess_and_split_text(df)
    df_chunks, index = create_faiss_index(df_chunks)
    
    st.success("File berhasil di upload.")

    # Simpan df_chunks dan index menggunakan session state
    st.session_state.df_chunks = df_chunks
    st.session_state.index = index

# Fungsi pencarian berdasarkan pertanyaan
def search(question: str, top_k: int = 5):
    if 'df_chunks' not in st.session_state or 'index' not in st.session_state:
        st.error("Silahkan upload file csv dahulu!")
        return []

    df_chunks = st.session_state.df_chunks
    index = st.session_state.index

    question_embedding = embedding_model.encode(question)
    distances, indices = index.search(np.array([question_embedding]), top_k)

    results = []
    for idx in indices[0]:
        result = {
            "title": df_chunks.iloc[idx]["title"],
            "chunk": df_chunks.iloc[idx]["chunk"]
        }
        results.append(result)
    return results

# Fungsi untuk mendapatkan jawaban
def get_answer(question: str):
    if 'df_chunks' not in st.session_state or 'index' not in st.session_state:
        st.error("Silahkan upload file csv dahulu!")
        return {}

    results = search(question, top_k=3)
    selected_chunks = " ".join([result["chunk"] for result in results])
    
    result = qa_pipeline({"question": question, "context": selected_chunks})
    return {"answer": result["answer"], "context_used": selected_chunks}

# Form untuk input pertanyaan
question = st.text_input("Tanyakan:")
if question:
    answer = get_answer(question)
    if answer:
        st.write(f"Jawaban: {answer['answer']}")
        st.write(f"Konteks: {answer['context_used']}")
