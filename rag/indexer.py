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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

CHUNK_SIZE = 300    # 每个chunk最大字符数
CHUNK_OVERLAP = 50  # 相邻chunk重叠字符数

def load_documents(kb_dir: str) -> list[dict]:
    """从目录中加载所有 txt/md 文档，返回 {{text, source}}"""
    docs = []
    for fname in os.listdir(kb_dir):
        if not fname.endswith((".txt", ".md")):
            continue
        fpath = os.path.join(kb_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        docs.append({"text": text, "source": fname})
    return docs

def split_chunks(text: str, source: str) -> list[dict]:
    """按字符数切分文本， 保留重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """批量获取 embedding 向量"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]

def build_index(kb_dir: str, index_path: str, meta_path: str):
    """完整索引构建流程"""
    docs = load_documents(kb_dir)
    print(f"[indexer] 加载文档 {len(docs)} 篇")

    all_chunks = []
    for doc in docs:
        chunks = split_chunks(doc["text"], doc["source"])
        all_chunks.extend(chunks)
    print(f"[indexer] 切分 chunk {len(all_chunks)} 个")

    texts = [c["text"] for c in all_chunks]
    embeddings = get_embeddings(texts)
    print(f"[indexer] 获取embedding 完成，维度={len(embeddings[0])}")

    vectors = np.array(embeddings, dtype="float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, index_path)
    print(f"[indexer] FAISS 索引已写入 {index_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"[indexer] Chunk 元数据已写入 {meta_path}")
