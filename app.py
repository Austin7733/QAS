import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')

# Load the pre-trained QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Define paths
file = 'arxiv_ml.csv'

# Load the dataset
df = pd.read_csv(file)

# Preprocessing text function
def preprocess_text(text):
    return text.replace('\n', ' ').replace('\r', '').strip()

# Apply preprocessing
df['abstract'] = df['abstract'].apply(preprocess_text)

# Tokenize text into chunks (for FAISS indexing)
def split_into_chunks(text, chunk_size=512):
    sentences = sent_tokenize(text)
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

# Prepare chunks data
chunks_data = []
for _, row in df.iterrows():
    article_id, title, chunks = row["id"], row["title"], row["chunks"]
    for i, chunk in enumerate(chunks):
        chunks_data.append({"article_id": article_id, "title": title, "chunk_id": f"{article_id}_{i}", "chunk": chunk})

chunks_df = pd.DataFrame(chunks_data)

# Create sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.array([model.encode(chunk) for chunk in chunks_df['chunk']])

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index for later use
faiss.write_index(index, 'faiss_index.bin')

# Query function for FAISS
def query_faiss(question, top_k=5):
    question_embedding = model.encode(question)
    distances, indices = index.search(np.array([question_embedding]), top_k)
    return [{"title": chunks_df.iloc[idx]["title"], "chunk": chunks_df.iloc[idx]["chunk"], "chunk_id": chunks_df.iloc[idx]["chunk_id"]} for idx in indices[0]]

# Function to answer a question using FAISS and the QA model
def answer_question(question, top_k=5):
    relevant_chunks = query_faiss(question, top_k)
    answers = []
    for chunk in relevant_chunks:
        qa_result = qa_pipeline({"question": question, "context": chunk['chunk']})
        answers.append({
            "title": chunk['title'],
            "chunk_id": chunk['chunk_id'],
            "answer": qa_result['answer'],
            "score": qa_result['score'],
            "context": chunk['chunk']
        })
    return sorted(answers, key=lambda x: x['score'], reverse=True)

# Streamlit app interface
st.title("Machine Learning QA System")
st.write("Sistem ini membantu menjawab pertanyaan Anda berdasarkan artikel Machine Learning dari ArXiv.")

user_question = st.text_input("Masukkan pertanyaan Anda:")

if user_question:
    st.write(f"### Pertanyaan Anda: {user_question}")
    with st.spinner("Sedang mencari jawaban..."):
        results = answer_question(user_question)
    if results:
        st.write("### Jawaban:")
        for i, res in enumerate(results):
            st.write(f"{i + 1}. **{res['title']}**")
            st.write(f"Jawaban: {res['answer']} (Skor: {res['score']:.4f})")
            with st.expander("Lihat konteks"):
                st.write(res['context'])
    else:
        st.write("Tidak ada jawaban yang ditemukan.")
