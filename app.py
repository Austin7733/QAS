import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File, HTTPException
import nltk
import uvicorn
from typing import List

nltk.download("punkt")

app = FastAPI()

# Load pre-trained models
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global variables for FAISS index and dataframe
df_chunks = None
index = None

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global df_chunks, index
    try:
        df = pd.read_csv(file.file)

        # Preprocessing function
        def preprocess_text(text):
            return text.replace('\n', ' ').replace('\r', '').strip()

        df["abstract"] = df["abstract"].apply(preprocess_text)
        
        # Function to split text into chunks
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
        
        # Create dataframe for chunks
        chunks_list = []
        for idx, row in df.iterrows():
            for chunk in row["chunks"]:
                chunks_list.append({"title": row["title"], "chunk": chunk})

        df_chunks = pd.DataFrame(chunks_list)

        # Create sentence embeddings
        embeddings = np.array([embedding_model.encode(chunk) for chunk in df_chunks["chunk"]])
        
        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        return {"message": "File uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/")
def search(question: str, top_k: int = 5):
    global df_chunks, index
    if df_chunks is None or index is None:
        raise HTTPException(status_code=400, detail="No data available. Please upload a CSV file first.")
    
    # Convert question to embedding
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

@app.get("/answer/")
def get_answer(question: str):
    global df_chunks, index
    if df_chunks is None or index is None:
        raise HTTPException(status_code=400, detail="No data available. Please upload a CSV file first.")

    # Embedding pertanyaan
    question_embedding = embedding_model.encode(question)
    distances, indices = index.search(np.array([question_embedding]), 3)  # Ambil 3 top context

    # Gabungkan context terbaik
    selected_chunks = " ".join(df_chunks.iloc[idx]["chunk"] for idx in indices[0])

    # Berikan ke QA model
    result = qa_pipeline({"question": question, "context": selected_chunks})

    return {
        "answer": result["answer"],
        "context_used": selected_chunks
    }

@app.post("/evaluate/")
def evaluate_model(questions: List[str]):
    global df_chunks, index
    if df_chunks is None or index is None:
        raise HTTPException(status_code=400, detail="No data available. Please upload a CSV file first.")

    results = []
    for question in questions:
        question_embedding = embedding_model.encode(question)
        distances, indices = index.search(np.array([question_embedding]), 3)
        
        selected_chunks = " ".join(df_chunks.iloc[idx]["chunk"] for idx in indices[0])
        result = qa_pipeline({"question": question, "context": selected_chunks})

        results.append({
            "question": question,
            "answer": result["answer"],
            "context_used": selected_chunks
        })

    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
