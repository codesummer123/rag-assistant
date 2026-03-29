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

def load_index():
    """加载 FAISS 索引和 chunk 元数据"""
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

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
    index, chunks = load_index()

    query_vec = np.array([get_query_embedding(query)], dtype="float32")
    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({"text": chunk["text"], "source": chunk["source"], "score": float(dist)})
        return results