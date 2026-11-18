"""Print resident memory after each startup stage (used in CI to keep the app under 512 MB)."""
import os, sys
sys.path.insert(0, os.getcwd())

def rss():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) // 1024
    return -1

def stage(name):
    print(f"{rss():5d} MB  after {name}", flush=True)

stage("python")
import numpy  # noqa
stage("numpy")
import faiss  # noqa
stage("faiss")
import fastapi, httpx  # noqa
stage("fastapi+httpx")
import onnxruntime, tokenizers  # noqa
stage("onnxruntime+tokenizers")
import fastembed  # noqa
stage("fastembed import")
from rag.embeddings import get_embedder
emb = get_embedder()
stage("embedder loaded")
emb.embed_query("When is the handover of Neptune?")
stage("first query embedded")
from rag.store import Store
store = Store(emb)
stage("store loaded")
for q in ["cheapest villa in Jeddah", "ما هي مشاريع دار جلوبال في عُمان؟"]:
    store.search(q)
stage("searches")
