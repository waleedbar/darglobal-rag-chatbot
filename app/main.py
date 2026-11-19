from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Literal

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


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class Ask(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=12)


METRICS: Counter = Counter()
LATENCIES: deque = deque(maxlen=500)
STARTED = time.time()


def _keep_warm() -> None:
    url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PUBLIC_URL")
    until = os.getenv("KEEP_WARM_UNTIL", "")
    if not url or not until:
        return
    import httpx

    def loop() -> None:
        while dt.date.today() <= dt.date.fromisoformat(until):
            time.sleep(600)
            try:
                httpx.get(f"{url.rstrip('/')}/api/health", timeout=20)
            except httpx.HTTPError as e:
                log.warning("keep-warm ping failed: %s", e)
        log.info("keep-warm period ended (%s)", until)

    threading.Thread(target=loop, name="keep-warm", daemon=True).start()
    log.info("keep-warm on until %s", until)


_keep_warm()


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
    res = answer(body.question, store, llm, history=[t.model_dump() for t in body.history])
    res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    METRICS.update(questions=1, idk=res["idk"], cached=bool(res.get("cached")), degraded=bool(res.get("degraded")),
                   follow_ups=bool(body.history), **{f"lang_{res['lang']}": 1})
    if not res.get("cached"):
        LATENCIES.append(res["latency_ms"])
    log.info("q=%r lang=%s idk=%s model=%s sources=%d %dms", body.question[:80], res["lang"], res["idk"],
             res.get("model"), len(res["sources"]), res["latency_ms"])
    return res


@app.get("/api/health")
def health():
    return {"ok": True, "llm_configured": llm.configured, **store.stats}


@app.get("/api/metrics")
def metrics():
    lat = sorted(LATENCIES)
    pct = (lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None)
    return {"uptime_s": int(time.time() - STARTED), **METRICS,
            "latency_ms": {"p50": pct(0.5), "p90": pct(0.9), "n": len(lat)}, "models": getattr(llm, "models", [])}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
