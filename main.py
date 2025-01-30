import os
import pandas as pd
import numpy as np
import faiss
import streamlit as st
import nltk
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from multiprocessing import Pool, cpu_count

# Download NLTK tokenizer
nltk.download("punkt")

# Model dan pipeline
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def process_chunk(text):
    """Tokenize and generate embeddings for a text chunk."""
    sentences = nltk.sent_tokenize(text)
    return sentences

def batch_encode(chunks):
    """Parallel embedding encoding."""
    return embedding_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

def preprocess_and_split_text(df, chunk_size=500):
    """Optimized text preprocessing using chunking."""
    df["abstract"] = df["abstract"].astype(str).str.replace('\n', ' ').str.replace('\r', '').str.strip()
    df["chunks"] = df["abstract"].apply(process_chunk)
    return df

def create_faiss_index(df_chunks):
    """Batch process embeddings and build FAISS index."""
    all_chunks = [(row["title"], chunk) for _, row in df_chunks.iterrows() for chunk in row["chunks"]]
    df_chunks = pd.DataFrame(all_chunks, columns=["title", "chunk"])
    
    # Parallel embeddings
    with Pool(cpu_count()) as pool:
        embeddings = pool.map(batch_encode, np.array_split(df_chunks["chunk"], cpu_count()))
    embeddings = np.vstack(embeddings)
    
    # FAISS Index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return df_chunks, index

# Streamlit UI
st.title("Optimized Machine Learning Q&A")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], accept_multiple_files=False)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, chunksize=10000)
    df = pd.concat(df)  # Merge chunks for processing
    df_chunks = preprocess_and_split_text(df)
    df_chunks, index = create_faiss_index(df_chunks)
    
    st.success("File processed successfully!")
    st.session_state.df_chunks = df_chunks
    st.session_state.index = index

def search(question, top_k=5):
    """Optimized search function using FAISS."""
    if 'df_chunks' not in st.session_state or 'index' not in st.session_state:
        st.error("Please upload a file first!")
        return []
    
    question_embedding = embedding_model.encode(question)
    distances, indices = st.session_state.index.search(np.array([question_embedding]), top_k)
    results = [{"title": st.session_state.df_chunks.iloc[idx]["title"], "chunk": st.session_state.df_chunks.iloc[idx]["chunk"]} for idx in indices[0]]
    return results

def get_answer(question):
    """Retrieve an answer from indexed chunks."""
    if 'df_chunks' not in st.session_state or 'index' not in st.session_state:
        st.error("Please upload a file first!")
        return {}
    
    results = search(question, top_k=3)
    selected_chunks = " ".join([result["chunk"] for result in results])
    result = qa_pipeline({"question": question, "context": selected_chunks})
    return {"answer": result["answer"], "context_used": selected_chunks}

question = st.text_input("Ask a question:")
if question:
    answer = get_answer(question)
    if answer:
        st.write(f"Answer: {answer['answer']}")
        st.write(f"Context used: {answer['context_used']}")
