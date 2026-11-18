"""Build and load the hybrid index: FAISS (dense, cosine) + BM25 (keyword), fused with RRF."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import chunk_doc
from .summaries import build_summaries
from .text import tokenize

log = logging.getLogger("rag.store")
ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.getenv("RAW_DIR", ROOT / "data" / "raw"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "index"))


def load_docs(raw_dir: Path = RAW) -> list[dict]:
    docs = []
    for f in sorted(raw_dir.glob("*.jsonl")):
        docs += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    return docs


def build(embedder, raw_dir: Path = RAW, out: Path = INDEX_DIR) -> dict:
    docs = load_docs(raw_dir)
    docs += build_summaries(docs)
    chunks = [c for d in docs for c in chunk_doc(d)]
    if not chunks:
        raise SystemExit(f"no documents found in {raw_dir}")
    log.info("embedding %d chunks from %d docs with %s", len(chunks), len(docs), embedder.name)
    vecs = embedder.embed_passages([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])  # vectors are L2-normalised -> inner product == cosine
    index.add(vecs)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "vectors.faiss"))
    (out / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    stats = {"docs": len(docs), "chunks": len(chunks), "dim": int(vecs.shape[1]), "embedder": embedder.name,
             "by_source": {s: sum(1 for d in docs if d["source"] == s) for s in {d["source"] for d in docs}},
             "scraped_at": max((d.get("scraped_at") or "" for d in docs), default="")}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


class Store:
    def __init__(self, embedder, index_dir: Path = INDEX_DIR):
        self.embedder = embedder
        self.index = faiss.read_index(str(index_dir / "vectors.faiss"))
        self.chunks = [json.loads(line) for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.stats = json.loads((index_dir / "stats.json").read_text())

    def search(self, query: str, k: int = 6, pool: int = 30) -> list[dict]:
        """Hybrid search. Returns chunks with `score` (cosine of best dense match) and `rrf`."""
        qv = self.embedder.embed_query(query)
        sims, ids = self.index.search(qv, min(pool, len(self.chunks)))
        dense_rank = {int(i): r for r, i in enumerate(ids[0]) if i >= 0}
        dense_sim = {int(i): float(s) for s, i in zip(sims[0], ids[0]) if i >= 0}
        bm = self.bm25.get_scores(tokenize(query))
        top_bm = np.argsort(-bm)[:pool]
        bm_rank = {int(i): r for r, i in enumerate(top_bm) if bm[i] > 0}
        fused: dict[int, float] = {}
        for ranks in (dense_rank, bm_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        out = []
        for i in best:
            c = dict(self.chunks[i])
            c["rrf"] = round(fused[i], 5)
            c["score"] = round(dense_sim.get(i, float(self.index.reconstruct(i) @ qv[0])), 4)
            out.append(c)
        return out
