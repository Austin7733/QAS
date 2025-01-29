
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st
import nltk

nltk.download('punkt_tab')

# Load pre-trained QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Load data (use Streamlit's uploader for CSV file)
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocessing function
    def preprocess_text(text):
        return text.replace('\n', ' ').replace('\r', '').strip()

    df['abstract'] = df['abstract'].apply(preprocess_text)

    # Define chunking and embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def split_into_chunks(text, chunk_size=512):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    df['chunks'] = df['abstract'].apply(split_into_chunks)

    # Create sentence embeddings
    embeddings = np.array([model.encode(chunk) for chunk in df['chunks'].explode()])

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Function to query FAISS
    def query_faiss(question, top_k=5):
        question_embedding = model.encode(question)
        distances, indices = index.search(np.array([question_embedding]), top_k)
        return [{"title": df.iloc[idx]["title"], "chunk": df.iloc[idx]["chunks"], "chunk_id": df.iloc[idx]["id"]} for idx in indices[0]]

    # Function to answer question using the QA model
    def get_answer(question, context):
        result = qa_pipeline({"question": question, "context": context})
        return result["answer"]

    # Streamlit interface
    st.title("Machine Learning QA System")
    st.markdown("### Tanyakan tentang topik Machine Learning!")

    context = st.text_area("Masukkan konteks artikel Machine Learning:", "")
    question = st.text_input("Masukkan pertanyaan Anda:", "")

    if st.button("Cari Jawaban"):
        if context and question:
            answer = get_answer(question, context)
            st.write(f"**Jawaban:** {answer}")
        else:
            st.write("Masukkan konteks dan pertanyaan terlebih dahulu!")
