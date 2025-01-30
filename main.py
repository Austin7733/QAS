import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st
import nltk
from tqdm import tqdm

# Download NLTK tokenizer
nltk.download("punkt")

# Model dan pipeline
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Fungsi untuk memproses dan membagi teks lebih efisien
def preprocess_and_split_text(df, chunk_size=2):
    df["abstract"] = df["abstract"].str.replace('\n', ' ').str.replace('\r', '').str.strip()
    df["chunks"] = df["abstract"].apply(lambda x: nltk.sent_tokenize(x))
    
    # Memecah menjadi chunk kecil agar lebih cepat diproses
    df_expanded = []
    for _, row in df.iterrows():
        sentences = row["chunks"]
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i+chunk_size])
            df_expanded.append({"title": row["title"], "chunk": chunk})
    
    return pd.DataFrame(df_expanded)

# Fungsi untuk membuat embeddings dan index FAISS lebih cepat
def create_faiss_index(df_chunks):
    embeddings = embedding_model.encode(df_chunks["chunk"].tolist(), batch_size=32, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)  # HNSW untuk pencarian lebih cepat
    index.add(embeddings)
    return df_chunks, index

# Streamlit interface
st.title("Apa yang ingin anda ketahui tentang Machine Learning?")

uploaded_file = st.file_uploader("Silahkan upload file csv!", type=["csv"], accept_multiple_files=False)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_chunks = preprocess_and_split_text(df)
    df_chunks, index = create_faiss_index(df_chunks)
    
    st.success("File berhasil di upload dan diproses.")
    
    # Simpan ke session state
    st.session_state.df_chunks = df_chunks
    st.session_state.index = index

# Fungsi pencarian berdasarkan pertanyaan
def search(question: str, top_k: int = 5):
    if 'df_chunks' not in st.session_state or 'index' not in st.session_state:
        st.error("Silahkan upload file csv dahulu!")
        return []
    
    df_chunks = st.session_state.df_chunks
    index = st.session_state.index
    
    question_embedding = embedding_model.encode([question], convert_to_numpy=True)
    distances, indices = index.search(question_embedding, top_k)
    
    results = [{"title": df_chunks.iloc[idx]["title"], "chunk": df_chunks.iloc[idx]["chunk"]} for idx in indices[0]]
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
