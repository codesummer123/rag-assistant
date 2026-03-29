from fastapi import FastAPI, HTTPException
import time
from app.schemas import ChatRequest, ChatResponse
from rag.retriever import retrieve
from rag.generator import generate

app = FastAPI(title="RAG Assistant", version="0.1.0")

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    start = time.time()

    contexts = retrieve(req.query, k=3)
    answer = generate(req.query, contexts)

    latency_ms = int((time.time() - start) * 1000)
    citations = list({c["source"] for c in contexts})

    return ChatResponse(
        answer=answer,
        citations=citations,
        latency_ms=latency_ms
    )
