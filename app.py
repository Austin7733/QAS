import os
import faiss
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Download tokenizer
nltk.download('punkt')

# Streamlit App Title
st.title("Machine Learning QA System")
st.markdown("### Tanyakan tentang topik Machine Learning!")

# Load pre-trained QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Membaca CSV dengan Optimasi**
    df = pd.read_csv(uploaded_file, dtype={"id": str}, low_memory=False).fillna("")

    # Preprocessing Cepat**
    df['abstract'] = df['abstract'].str.replace(r'[\n\r]', ' ', regex=True).str.strip()

    # Fungsi Chunking yang Lebih Cepat**
    def fast_chunking(text, chunk_size=512):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    df['chunks'] = df['abstract'].apply(fast_chunking)

    # Menghasilkan Embedding Secara Paralel**
    def encode_batch(batch):
        return embedding_model.encode(batch, convert_to_numpy=True, batch_size=64)

    batch_size = 64
    data_batches = np.array_split(df['chunks'].explode().tolist(), len(df) // batch_size)
    embeddings = []

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(encode_batch, data_batches))

    embeddings = np.vstack(results)

    # Simpan dan Load FAISS Index**
    index_file = "faiss_index.bin"
    
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        st.write("📥 FAISS Index Loaded from Cache")
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, index_file)
        st.write("📤 FAISS Index Created and Saved")

    # Query FAISS**
    def query_faiss(question, top_k=5):
        question_embedding = embedding_model.encode(question)
        distances, indices = index.search(np.array([question_embedding]), top_k)
        return [{"title": df.iloc[idx]["title"], "chunk": df.iloc[idx]["chunks"]} for idx in indices[0]]

    # Fungsi Menjawab Pertanyaan**
    def get_answer(question, context):
        result = qa_pipeline({"question": question, "context": context})
        return result["answer"]

    # Streamlit UI
    context = st.text_area("Masukkan konteks artikel Machine Learning:", "")
    question = st.text_input("Masukkan pertanyaan Anda:", "")

    if st.button("Cari Jawaban"):
        if context and question:
            answer = get_answer(question, context)
            st.write(f"**Jawaban:** {answer}")
        else:
            st.write("Masukkan konteks dan pertanyaan terlebih dahulu!")
