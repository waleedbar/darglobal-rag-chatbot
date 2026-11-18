"""Embedding backends.

* FastEmbedEmbedder - production: intfloat/multilingual-e5-large (ONNX, CPU), strong for Arabic and
  English and for cross-lingual retrieval (Arabic question -> English page).
* HashEmbedder - deterministic offline fallback used in unit tests (no model download).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from .text import normalize, tokenize

DEFAULT_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class FastEmbedEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | None = None):
        from fastembed import TextEmbedding  # imported lazily: heavy

        self.name = model
        self.model = TextEmbedding(model_name=model, cache_dir=cache_dir or os.getenv("MODEL_CACHE", "models"))
        self.e5 = "e5" in model.lower()

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        texts = [("passage: " + t) if self.e5 else t for t in texts]
        return _l2(np.asarray(list(self.model.embed(texts, batch_size=16)), dtype="float32"))

    def embed_query(self, text: str) -> np.ndarray:
        q = ("query: " + text) if self.e5 else text
        return _l2(np.asarray(list(self.model.embed([q])), dtype="float32"))


class HashEmbedder:
    """Bag-of-words hashing into a fixed space; good enough to exercise the pipeline in tests."""

    name = "hash-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in tokenize(normalize(text)):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _l2(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._vec(text)[None, :])


def get_embedder():
    return HashEmbedder() if os.getenv("EMBED_BACKEND") == "hash" else FastEmbedEmbedder()
