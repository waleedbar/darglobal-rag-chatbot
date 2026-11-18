"""Embedding backends.

* FastEmbedEmbedder - production: paraphrase-multilingual-MiniLM-L12-v2 (ONNX, quantized to int8 at build time, 50+ languages
  incl. Arabic) so the container fits a 512 MB free instance; set EMBED_MODEL=intfloat/multilingual-e5-large
  for higher recall when more RAM is available (both run locally: no API quota is spent on embeddings).
* HashEmbedder - deterministic offline fallback used in unit tests (no model download).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .text import normalize, tokenize

DEFAULT_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def _trim_heap() -> None:
    """Give memory freed while loading the model back to the OS (glibc keeps it otherwise)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class FastEmbedEmbedder:
    """Multilingual sentence embeddings with onnxruntime.

    fastembed is used once to download the ONNX model into the cache; inference then runs the
    (build-time int8-quantized) graph directly with graph optimisations off. ORT's constant folding
    would otherwise de-quantize the embedding table back to float32 (~570 MB resident instead of
    ~150 MB), which does not fit a 512 MB free instance.
    """

    MAX_TOKENS = 128

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | None = None):
        self.name = model
        self.e5 = "e5" in model.lower()
        cache = Path(cache_dir or os.getenv("MODEL_CACHE", "models"))
        found = self._find(cache)
        if found is None:  # first run (image build): let fastembed download the files
            from fastembed import TextEmbedding

            TextEmbedding(model_name=model, cache_dir=str(cache), lazy_load=True)
            found = self._find(cache)
        if found is None:
            raise RuntimeError(f"embedding model {model} not found in {cache}")
        import onnxruntime as ort
        from tokenizers import Tokenizer

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.enable_cpu_mem_arena = False
        so.intra_op_num_threads = int(os.getenv("EMBED_THREADS", "2"))
        so.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(str(found), so, providers=["CPUExecutionProvider"])
        except Exception:  # un-quantized float16 file: needs ORT's default graph transforms
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(found), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        _trim_heap()
        self.tok = Tokenizer.from_file(str(found.parent / "tokenizer.json"))
        self.tok.enable_truncation(self.MAX_TOKENS)
        pad = self.tok.token_to_id("<pad>")
        self.tok.enable_padding(pad_id=1 if pad is None else pad, pad_token="<pad>")

    def _find(self, cache: Path) -> Path | None:
        key = self.name.split("/")[-1].lower()
        hits = [p for p in cache.rglob("*.onnx") if key in str(p).lower() and (p.parent / "tokenizer.json").exists()]
        hits.sort(key=lambda p: (p.name != "model_int8.onnx", str(p)))  # prefer the build-time int8 model
        return hits[0] if hits else None

    def _embed(self, texts: list[str], batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            ids = np.array([e.ids for e in enc], dtype="int64")
            mask = np.array([e.attention_mask for e in enc], dtype="int64")
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0].astype("float32")
            if hidden.ndim == 3:  # mean pooling over real tokens
                m = mask[..., None].astype("float32")
                hidden = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out.append(hidden.astype("float32"))
        return _l2(np.concatenate(out))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed([("passage: " + t) if self.e5 else t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([("query: " + text) if self.e5 else text])


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
