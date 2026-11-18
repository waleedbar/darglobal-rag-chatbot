# syntax=docker/dockerfile:1.7
# Bilingual RAG chatbot over public Dar Global + Wasalt data.
# The embedding model and the vector index are baked into the image at build time,
# so the container starts fast and needs no network except calls to OpenRouter.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE=/app/models HF_HOME=/app/models PORT=7860

# EMBED_BACKEND=hash builds a tiny offline image for CI/local smoke tests (no model download)
ARG EMBED_BACKEND=fastembed
ARG EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV EMBED_BACKEND=${EMBED_BACKEND} EMBED_MODEL=${EMBED_MODEL}

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .

# Download the multilingual embedding model once (ONNX) so runtime needs no model download,
# then quantize it to int8 so the whole service fits in 512 MB of RAM.
RUN if [ "$EMBED_BACKEND" != "hash" ]; then \
      python -c "import os; from fastembed import TextEmbedding; TextEmbedding(os.environ['EMBED_MODEL'], cache_dir=os.environ['MODEL_CACHE'], lazy_load=True)" && \
      pip install "onnx>=1.17" && python scripts/quantize_model.py && pip uninstall -y onnx; fi

# Decrypt the Wasalt snapshot if the DATA_KEY build secret is provided (not redistributed in plain form),
# then embed every chunk and write the FAISS + BM25 index.
RUN --mount=type=secret,id=DATA_KEY,mode=0444,required=false \
    DATA_KEY="$(cat /run/secrets/DATA_KEY 2>/dev/null || true)" python -m scraper.wasalt_import decrypt && \
    python -m rag.build_index && rm -f data/raw/wasalt.jsonl data/private/*.jsonl && \
    chown -R user:user /app

USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/api/health', timeout=4)"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --no-access-log"]
