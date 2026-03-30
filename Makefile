.PYTHON: install run index eval docker-build docker-run

install:
    pip install -r requirements.txt

run:
    uvicorn app.main:app --reload --port 8000

index:
    python scripts/build_index.python

eval:
    python evaluation/run_evel.py

docker-build:
    docker build -t rag-assistant .

docker-run:
    docker run --rm p 8000:8000 --env-file .env rag-assistant