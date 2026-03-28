# RAG Assistant

A practical Retrieval-Augmented Generation (RAG) project focused on Chinese enterprise knowledge base scenarios.

Current objective for this stage is to establish a reliable MVP pipeline:

1. Document ingestion and chunking
2. Vector retrieval
3. Answer generation with citations
4. Basic evaluation and observability

## Current Status

- ✅ FastAPI service initialized
- ✅ Health check endpoint (`/health`)
- 🚧 Indexing pipeline (in progress)
- 🚧 Retrieval pipeline (in progress)
- 🚧 Chat endpoint with citations (planned)
- 🚧 Baseline evaluation script (planned)

## Tech Stack

- **Backend**: FastAPI
- **Language**: Python 3.11+
- **Validation**: Pydantic
- **Containerization**: Docker
- **RAG Components (planned)**: Embeddings + FAISS + Prompted generation

## Project Structure

```text
app/                  # API layer
  main.py             # FastAPI entrypoint
  schemas.py          # Request/response models
rag/                  # Core RAG logic
  indexer.py
  retriever.py
  generator.py
data/kb/              # Knowledge base source files
evaluation/           # Evaluation dataset/scripts
scripts/              # Helper scripts (e.g., index build)
.env.example          # Environment variable template
Dockerfile
requirements.txt
```

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure environment

```bash
# Linux / macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Fill required values in `.env`.

### 3) Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

### 4) Verify service

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"OK"}
```

## Docker

```bash
docker build -t rag-assistant .
docker run --rm -p 8000:8000 rag-assistant
```

## Development Roadmap

- **v0.1.0**: service skeleton + health check
- **v0.2.0**: text chunking + vector index build
- **v0.3.0**: retrieval + `/chat` endpoint with citations
- **v0.4.0**: baseline evaluation + response metrics

## License

MIT. See `LICENSE` for details.
