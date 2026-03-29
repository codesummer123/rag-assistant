import sys
import os

# 让脚本在项目根目录运行时能找到rag包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.indexer import build_index

KB_DIR = "data/kb"
INDEX_DIR = "data/index.faiss"
META_PATH = "data/chunks_meta.json"

if __name__ == "__main__":
    build_index(KB_DIR, INDEX_DIR, META_PATH)
    print(f"[build_index] 完成")