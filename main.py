import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk
import streamlit as st
from typing import List

# Download NLTK tokenizer
nltk.download("punkt_tab")

# Load pre-trained models
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global variables for FAISS index and dataframe
df_chunks = None
index = None

# Function to upload and process CSV files
def upload_files():
    global df_chunks, index
    
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files:
        all_chunks = []
        
        for file in uploaded_files:
            df = pd.read_csv(file)
            
            # Preprocessing text
            def preprocess_text(text):
                return text.replace('\n', ' ').replace('\r', '').strip()
            
            df["abstract"] = df["abstract"].apply(preprocess_text)
            
            # Splitting text into chunks
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
            
            df["chunks"] = df["abstract"].apply(split_into_chunks)
            
            # Storing chunks into a list
            for _, row in df.iterrows():
                for chunk in row["chunks"]:
                    all_chunks.append({"title": row["title"], "chunk": chunk})
        
        # Creating dataframe for all text chunks
        df_chunks = pd.DataFrame(all_chunks)
        
        # Convert text to embeddings
        embeddings = np.array([embedding_model.encode(chunk) for chunk in df_chunks["chunk"]])
        
        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        st.success(f"{len(uploaded_files)} files uploaded and processed successfully")

# Function to search for relevant chunks based on the question
def search(question: str, top_k: int = 5):
    global df_chunks, index
    
    if df_chunks is None or index is None:
        st.error("No data available. Please upload files first.")
        return
    
    # Encode the question into embeddings
    question_embedding = embedding_model.encode(question)
    
    # Perform FAISS search
    distances, indices = index.search(np.array([question_embedding]), top_k)
    
    # Get the search results
    results = []
    for idx in indices[0]:
        result = {
            "title": df_chunks.iloc[idx]["title"],
            "chunk": df_chunks.iloc[idx]["chunk"]
        }
        results.append(result)
    
    return results

# Function to get an answer from the most relevant text chunks
def get_answer(question: str):
    global df_chunks, index
    
    if df_chunks is None or index is None:
        st.error("No data available. Please upload files first.")
        return
    
    # Encode the question into embeddings
    question_embedding = embedding_model.encode(question)
    
    # Perform FAISS search to find the best matching text chunks
    distances, indices = index.search(np.array([question_embedding]), 3)
    
    # Combine the relevant text chunks
    selected_chunks = " ".join(df_chunks.iloc[idx]["chunk"] for idx in indices[0])
    
    # Use the QA model to answer the question
    result = qa_pipeline({"question": question, "context": selected_chunks})
    
    return {
        "answer": result["answer"],
        "context_used": selected_chunks
    }

# Streamlit App UI
st.title("Question Answering with CSV Data")

# Upload files section
upload_files()

# Input for asking a question
question = st.text_input("Ask a question:")

if question:
    # Perform search and display results
    results = search(question)
    if results:
        st.subheader("Relevant Chunks:")
        for result in results:
            st.write(f"**Title:** {result['title']}")
            st.write(f"**Chunk:** {result['chunk'][:500]}...")  # Display the first 500 characters of the chunk

        # Get the answer to the question
        answer = get_answer(question)
        if answer:
            st.subheader("Answer:")
            st.write(answer["answer"])
            st.subheader("Context Used for Answer:")
            st.write(answer["context_used"][:500])  # Display the first 500 characters of context
