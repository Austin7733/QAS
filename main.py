import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File, HTTPException
import nltk
import uvicorn
from typing import List

# Download NLTK tokenizer
nltk.download("punkt_tab")

app = FastAPI()

# Load pre-trained models
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global variables for FAISS index and dataframe
df_chunks = None
index = None

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    """Mengunggah dan memproses file CSV untuk indexing dengan FAISS"""
    global df_chunks, index
    try:
        if len(files) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 files allowed.")
        
        all_chunks = []
        
        for file in files:
            df = pd.read_csv(file.file)
            
            # Preprocessing teks
            def preprocess_text(text):
                return text.replace('\n', ' ').replace('\r', '').strip()
            
            df["abstract"] = df["abstract"].apply(preprocess_text)
            
            # Membagi teks abstrak menjadi potongan (chunks)
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
            
            # Menyimpan potongan teks ke dalam list
            for _, row in df.iterrows():
                for chunk in row["chunks"]:
                    all_chunks.append({"title": row["title"], "chunk": chunk})
        
        # Membuat dataframe untuk semua potongan teks
        df_chunks = pd.DataFrame(all_chunks)
        
        # Mengubah teks menjadi vektor embeddings
        embeddings = np.array([embedding_model.encode(chunk) for chunk in df_chunks["chunk"]])
        
        # Membuat FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        return {"message": f"{len(files)} files uploaded and processed successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/")
def search(question: str, top_k: int = 5):
    """Mencari potongan teks yang paling relevan berdasarkan query pengguna"""
    global df_chunks, index
    if df_chunks is None or index is None:
        raise HTTPException(status_code=400, detail="No data available. Please upload files first.")
    
    # Encode pertanyaan ke dalam vektor
    question_embedding = embedding_model.encode(question)
    
    # Melakukan pencarian FAISS
    distances, indices = index.search(np.array([question_embedding]), top_k)
    
    # Mengambil hasil pencarian
    results = []
    for idx in indices[0]:
        result = {
            "title": df_chunks.iloc[idx]["title"],
            "chunk": df_chunks.iloc[idx]["chunk"]
        }
        results.append(result)
    
    return results

@app.get("/answer/")
def get_answer(question: str):
    """Menjawab pertanyaan dengan model QA berdasarkan teks yang paling relevan"""
    global df_chunks, index
    if df_chunks is None or index is None:
        raise HTTPException(status_code=400, detail="No data available. Please upload files first.")
    
    # Encode pertanyaan ke dalam vektor
    question_embedding = embedding_model.encode(question)
    
    # Melakukan pencarian FAISS untuk menemukan potongan teks terbaik
    distances, indices = index.search(np.array([question_embedding]), 3)
    
    # Menggabungkan teks dari hasil pencarian
    selected_chunks = " ".join(df_chunks.iloc[idx]["chunk"] for idx in indices[0])
    
    # Menggunakan model QA untuk menjawab pertanyaan
    result = qa_pipeline({"question": question, "context": selected_chunks})
    
    return {
        "answer": result["answer"],
        "context_used": selected_chunks
    }

if __name__ == "__main__":
    # Membaca port dari Railway
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
