from fastapi import FastAPI

app = FastAPI(title="CN Enterprise RAG Assistant", version="0.1.0")

@app.get("/health")
def health_check():
    return {"status": "OK"}