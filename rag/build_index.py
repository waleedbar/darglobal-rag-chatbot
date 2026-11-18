"""Build the vector + keyword index from data/raw/*.jsonl.   python -m rag.build_index"""
import json
import logging

from .embeddings import get_embedder
from .store import build

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(build(get_embedder()), indent=2))
