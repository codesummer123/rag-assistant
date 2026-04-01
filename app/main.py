from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import time
from app.schemas import ChatRequest, ChatResponse
from rag.reranker import rerank
from rag.retriever import retrieve, init_retriever
from rag.generator import generate
from rag.rewriter import rewrite_query

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_retriever()
    yield

app = FastAPI(title="RAG Assistant", version="0.2.1", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    start = time.time()

    # 1. 口语提取关键词
    search_query = rewrite_query(req.query)
    print(f"[chat] 原始问题：{req.query} -> 修改后：{search_query}")

    # 2. 粗排检索
    initial_contexts = retrieve(search_query)

    # 3. 精排
    final_contexts = rerank(req.query, initial_contexts, top_n=3)

    # 4. 生成回答
    answer = generate(req.query, final_contexts)

    latency_ms = int((time.time() - start) * 1000)
    citations = list(c["source"] for c in final_contexts)

    return ChatResponse(
        answer=answer,
        citations=citations,
        latency_ms=latency_ms
    )
