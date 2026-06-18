from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import uuid

from services.pdf_processor import extract_and_chunk
from services.vector_store import VectorStore
from services.rag_chain import get_answer

app = FastAPI(title="PaperMind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = VectorStore()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.get("/")
def root():
    return {"status": "PaperMind API is running"}


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")

        content = await file.read()
        doc_id = str(uuid.uuid4())[:8]

        try:
            chunks = extract_and_chunk(content, file.filename, doc_id)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not read {file.filename}: {str(e)}")

        store.add_documents(chunks, doc_id, file.filename)
        results.append({
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks": len(chunks)
        })

    return {"uploaded": results}


@app.get("/documents")
def list_documents():
    return {"documents": store.list_documents()}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    store.delete_document(doc_id)
    return {"message": f"Deleted {doc_id}"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if store.count() == 0:
        raise HTTPException(status_code=400, detail="Upload at least one PDF first")

    chunks = store.search(req.question, k=5)
    answer, sources = get_answer(req.question, chunks)
    return QueryResponse(answer=answer, sources=sources)


@app.delete("/reset")
def reset_store():
    store.reset()
    return {"message": "All documents cleared"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
