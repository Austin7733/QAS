# -*- coding: utf-8 -*-
"""QAS-RAG.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11ZBvPkDQOfPb7tbZeIwd9W6d1ZH3zIzT
"""

!pip install swifter
import pandas as pd
import swifter
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
import streamlit as st
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing

nltk.download('punkt_tab')

# 1. Load data
file = '/content/drive/MyDrive/BCC/arxiv_ml.csv'

df = pd.read_csv(file)
df.head()

# 2. Preprocessing teks
def preprocess_text(text): # Hapus karakter khusus seperti newline dan spasi tambahan
    text = text.replace('\n', ' ').replace('\r', '').strip()
    return text

# Terapkan preprocess text ke abstract
df['abstract'] = df['abstract'].apply(preprocess_text)
df.head()

df["chunks"] = df["abstract"].swifter.apply(split_into_chunks)

# Ubah format ke DataFrame yang sesuai
chunks_data = []
for _, row in df.iterrows():
    article_id, title, chunks = row["id"], row["title"], row["chunks"]
    for i, chunk in enumerate(chunks):
        chunks_data.append({"article_id": article_id, "title": title, "chunk_id": f"{article_id}_{i}", "chunk": chunk})

chunks_df = pd.DataFrame(chunks_data)
chunks_df.to_csv('processed_chunks.csv', index=False)

# 4. Membuat Embedding dengan Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Menghasilkan embedding untuk semua chunks...")

# Optimasi proses embedding dengan batch encoding
def encode_batch(batch):
    return model.encode(batch, convert_to_numpy=True, batch_size=32)

# Bagi data menjadi batch
batch_size = 32
data_batches = [chunks_df['chunk'][i:i+batch_size].tolist() for i in range(0, len(chunks_df), batch_size)]
embeddings = []

# Proses encoding dalam loop agar lebih stabil dengan tqdm sebagai progress bar
for batch in tqdm(data_batches, desc="Encoding chunks"):
    embeddings.append(encode_batch(batch))
embeddings = np.vstack(embeddings)

# 5. Membuat Index FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, 'faiss_index.bin')

# Simpan DataFrame tanpa embedding untuk referensi
chunks_df.to_csv('chunks_with_metadata.csv', index=False)

# 6. Query ke FAISS
def query_faiss(question, top_k=5):
    """Mencari potongan teks paling relevan berdasarkan pertanyaan pengguna."""
    question_embedding = model.encode(question, convert_to_numpy=True)
    distances, indices = index.search(np.array([question_embedding]), top_k)
    return [{"title": chunks_df.iloc[idx]["title"], "chunk": chunks_df.iloc[idx]["chunk"], "chunk_id": chunks_df.iloc[idx]["chunk_id"]} for idx in indices[0] if idx != -1]

# 7. Integrasi dengan Model QA
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_question(question, top_k=5):
    """Mengintegrasikan retrieval dengan model QA untuk jawaban lebih akurat."""
    relevant_chunks = query_faiss(question, top_k=top_k)
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

# 8. Streamlit App
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

results = query_faiss("What is reinforcement learning?", top_k=5)
for res in results:
    print(f"Title: {res['title']}\nChunk ID: {res['chunk_id']}\nChunk: {res['chunk']}\n")

qa_results = answer_question("What is reinforcement learning?", top_k=3)
for res in qa_results:
    print(f"Title: {res['title']}\nChunk ID: {res['chunk_id']}\nAnswer: {res['answer']}\nScore: {res['score']:.4f}\n")

!pip install swifter

from google.colab import files
uploaded = files.upload()

!pip install streamlit

import streamlit as st
from transformers import pipeline

# Muat model yang telah dilatih atau gunakan model pretrained
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

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



