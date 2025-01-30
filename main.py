import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import streamlit as st
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize
import nltk

# Download data tokenisasi NLTK
nltk.download('punkt')

# ========== LOAD MODELS ==========
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# ========== FUNGSI PEMROSESAN ==========

# Fungsi tokenisasi menggunakan NLTK
def tokenize_sentences(text):
    return word_tokenize(text)

def preprocess_and_split_text(df):
    df["abstract"] = df["abstract"].astype(str).str.replace('\n', ' ').str.replace('\r', '').str.strip()
    df["chunks"] = df["abstract"].apply(tokenize_sentences)
    return df

def batch_encode(chunks, batch_size=64):
    def encode_batch(batch):
        return embedding_model.encode(batch, show_progress_bar=False)
    
    embeddings = Parallel(n_jobs=-1)(delayed(encode_batch)(chunks[i:i + batch_size]) for i in range(0, len(chunks), batch_size))
    return np.vstack(embeddings)

# ========== FAISS INDEX HANDLING ==========

FAISS_INDEX_PATH = "faiss.index"

@st.cache_resource
def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    return None

def create_faiss_index(df_chunks):
    all_chunks = [(row["title"], chunk) for _, row in df_chunks.iterrows() for chunk in row["chunks"]]
    df_chunks = pd.DataFrame(all_chunks, columns=["title", "chunk"])
    
    embeddings = batch_encode(df_chunks["chunk"].tolist())
    
    d = embeddings.shape[1]
    if len(embeddings) > 100:  # Pastikan cukup data untuk IVFFlat
        nlist = 100
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(embeddings)
    else:
        index = faiss.IndexFlatL2(d)  # Pakai IndexFlat jika data kecil
    
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    return df_chunks, index

# ========== STREAMLIT INTERFACE ==========

st.title("Apa yang ingin Anda ketahui tentang Machine Learning?")

uploaded_file = st.file_uploader("Silahkan upload file CSV!", type=["csv"])

if uploaded_file:
    st.info("Memproses file, harap tunggu...")
    
    # Menangani upload CSV
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)  # Menggunakan Pandas untuk membaca file
        df_chunks = preprocess_and_split_text(df)
        
        # Load or create FAISS index
        index = load_faiss_index()
        if index is None:
            df_chunks, index = create_faiss_index(df_chunks)
        
        st.success("File berhasil di-upload dan diproses!")
        st.session_state.df_chunks = df_chunks
        st.session_state.index = index
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")

# ========== PENCARIAN ==========

def search(question: str, top_k: int = 5):
    if 'df_chunks' not in st.session_state or 'index' not in st.session_state:
        st.error("Silahkan upload file CSV dahulu!")
        return []

    df_chunks = st.session_state.df_chunks
    index = st.session_state.index

    question_embedding = embedding_model.encode(question)
    distances, indices = index.search(np.array([question_embedding]), top_k)

    valid_indices = [idx for idx in indices[0] if 0 <= idx < len(df_chunks)]
    
    results = [{"title": df_chunks.iloc[idx]["title"], "chunk": df_chunks.iloc[idx]["chunk"]} for idx in valid_indices]
    return results

def get_answer(question: str):
    results = search(question, top_k=3)
    if not results:
        return {}

    selected_chunks = " ".join([result["chunk"] for result in results])
    
    result = qa_pipeline({"question": question, "context": selected_chunks})
    return {"answer": result["answer"], "context_used": selected_chunks}

# ========== FORM INPUT ==========

question = st.text_input("Tanyakan:")
if question:
    answer = get_answer(question)
    if answer:
        st.write(f"**Jawaban:** {answer['answer']}")
        st.write(f"**Konteks:** {answer['context_used']}")
