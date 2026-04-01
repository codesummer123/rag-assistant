import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

INDEX_PATH = "data/index.faiss"
META_PATH = "data/chunks_meta.json"

# 全局变量用于缓存
_cached_index = None
_cached_chunks = None

def init_retriever():
    """初始化检索器，将索引加载到内存"""
    global _cached_index, _cached_chunks

    if not os.path.exists(INDEX_PATH) or not os.path.isfile(META_PATH):
        print(f"[retriever] 警告：索引文件不存在，请先运行 python scripts/build_index.py")
        return

    print(f"[retriever] 正在从 {INDEX_PATH} 加载索引")
    _cached_index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        _cached_chunks = json.load(f)
    print(f"[retriever] 索引加载成功，包含{len(_cached_chunks)} 个chunk")

def get_query_embedding(query: str) -> list[float]:
    """将用户的问题转换成 embedding 向量"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding

def retrieve(query: str, k: int = 3) -> list[dict]:
    """
    检索相关的 Top-K 个 chunk
    返回：[{"text": ..., "source": ..., "score": ...}]
    """
    global _cached_index, _cached_chunks

    if _cached_index is None:
        init_retriever()
    if _cached_index is None:
        return []

    query_vec = np.array([get_query_embedding(query)], dtype="float32")
    distances, indices = _cached_index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(_cached_chunks):
            continue
        chunk = _cached_chunks[idx]
        results.append({"text": chunk["text"], "source": chunk["source"], "score": float(dist)})
    return results