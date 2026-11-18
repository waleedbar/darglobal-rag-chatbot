"""FastAPI app: chat UI at /, JSON API at /api/chat."""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.embeddings import get_embedder
from rag.llm import EchoLLM, OpenRouterLLM
from rag.pipeline import answer
from rag.store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")
STATIC = Path(__file__).with_name("static")
RATE = int(os.getenv("RATE_LIMIT_PER_MIN", "12"))

app = FastAPI(title="DarGlobal & Wasalt RAG chatbot", version="1.0")
store = Store(get_embedder())
llm = EchoLLM() if os.getenv("LLM_BACKEND") == "echo" else OpenRouterLLM()
_hits: dict[str, deque] = defaultdict(deque)


class Ask(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


def _rate_limit(ip: str) -> None:
    now, q = time.monotonic(), _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE:
        raise HTTPException(429, "Too many questions - please wait a minute.")
    q.append(now)


@app.post("/api/chat")
def chat(body: Ask, request: Request):
    ip = (request.headers.get("x-forwarded-for") or request.client.host or "?").split(",")[0].strip()
    _rate_limit(ip)
    t0 = time.perf_counter()
    res = answer(body.question, store, llm)
    res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    log.info("q=%r lang=%s idk=%s model=%s sources=%d %dms", body.question[:80], res["lang"], res["idk"],
             res.get("model"), len(res["sources"]), res["latency_ms"])
    return res


@app.get("/api/health")
def health():
    return {"ok": True, "llm_configured": llm.configured, **store.stats}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
