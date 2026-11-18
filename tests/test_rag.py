"""RAG pipeline tests with a fixed mini-corpus, the hashing embedder and fake LLMs (fully offline)."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from rag.chunking import split_text
from rag.embeddings import HashEmbedder
from rag.llm import EchoLLM, LLMError, QuotaExceeded
from rag.pipeline import CACHE, IDK, answer, retrieval_query
from rag.store import Store, build
from rag.summaries import build_summaries
from rag.text import detect_lang, tokenize

DOCS = [
    {"id": "dg:neptune", "source": "darglobal", "type": "project", "url": "https://darglobal.co.uk/neptune",
     "title": "Neptune, Interiors by Mouawad", "lang": "en", "meta": {"location": "Riyadh, KSA"},
     "text": "Project: Neptune, Interiors by Mouawad\nLocation: Riyadh, KSA\nHandover/completion: December 2027\n"
             "Five-bedroom branded villas, 300 - 448 sqm. Nearby: schools, mosques, parks."},
    {"id": "dg:marea", "source": "darglobal", "type": "project", "url": "https://darglobal.co.uk/marea",
     "title": "Marea, Interiors by Missoni", "lang": "en", "meta": {"location": "Cortesin, Spain"},
     "text": "Project: Marea, Interiors by Missoni\nLocation: Cortesin, Spain\nBeachfront villas on the Costa del Sol."},
] + [
    {"id": f"wasalt:{i}", "source": "wasalt", "type": "listing", "url": f"https://wasalt.sa/en/property/sale/villa-{i}",
     "title": f"Villa {i}", "lang": "en",
     "meta": {"city": "Jeddah", "subtype": "Villa", "purpose": "sale", "price": p, "currency": "SAR",
              "district": d, "category_page": "https://wasalt.sa/en/villas-for-sale-in-jeddah"},
     "text": f"Listing: Villa {i}\nLocation: {d}, Jeddah\nPrice: {p:,} SAR"}
    for i, (p, d) in enumerate([(1_200_000, "Al Basateen"), (3_500_000, "Al Shati"), (2_000_000, "Obhur")])
]


@pytest.fixture(scope="module")
def store():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "raw").mkdir()
    (tmp / "raw" / "docs.jsonl").write_text("\n".join(json.dumps(d) for d in DOCS))
    stats = build(HashEmbedder(), raw_dir=tmp / "raw", out=tmp / "index")
    # + statistics docs: cheapest / most expensive / price stats for "Villa" and "All types" in Jeddah
    assert stats["docs"] == len(DOCS) + 6
    return Store(HashEmbedder(), index_dir=tmp / "index")


class FakeLLM:
    configured = True

    def __init__(self, reply):
        self.reply, self.last_user, self.calls, self.history = reply, "", 0, None

    def chat(self, system, user, history=None):
        self.last_user, self.history = user, history
        self.calls += 1
        return self.reply, "fake/model:free"


@pytest.fixture(autouse=True)
def _fresh_cache():
    CACHE.data.clear()
    yield
    CACHE.data.clear()


def test_language_detection_and_arabic_tokens():
    assert detect_lang("متى موعد تسليم مشروع نبتون؟") == "ar"
    assert detect_lang("When is the Neptune handover?") == "en"
    assert detect_lang("كم سعر DG1؟") == "ar"
    assert "مشروع" in tokenize("المشروع") or "مشروع" in tokenize("مشروع")


def test_chunking_respects_size_and_keeps_text():
    text = "\n\n".join(f"## Section {i}\n" + "word " * 120 for i in range(6))
    chunks = split_text(text)
    assert all(len(c) <= 600 for c in chunks) and len(chunks) >= 6
    assert any("Section 5" in c for c in chunks[-3:]) and sum(c.count("word") for c in chunks) >= 6 * 120


def test_hybrid_retrieval_finds_the_right_project(store):
    hits = store.search("When is the handover of Neptune?", k=3)
    assert hits[0]["url"] == "https://darglobal.co.uk/neptune"


def test_price_summary_documents():
    s = {d["url"].rsplit("#", 1)[1]: d for d in build_summaries(DOCS)}
    assert "The cheapest is 1,200,000 SAR" in s["cheapest-villa"]["text"] and "أرخص فلل للبيع في جدة" in s["cheapest-villa"]["text"]
    assert "The most expensive is 3,500,000 SAR" in s["most-expensive-villa"]["text"]
    assert "median 2,000,000 SAR" in s["price-stats-villa"]["text"] and "sample of 3 Wasalt listings" in s["price-stats-villa"]["text"]


def test_aggregate_questions_find_the_statistics_docs(store):
    for q, frag in [("What is the cheapest villa in Jeddah?", "#cheapest-villa"), ("أغلى فيلا في جدة", "#most-expensive-villa"),
                    ("average villa price in Jeddah", "#price-stats-villa")]:
        assert any(frag in h["url"] for h in store.search(q, k=3)), q


def test_answer_with_citations_renumbered(store):
    llm = FakeLLM("Handover is December 2027 [2]. Villas are five-bedroom [2][2].")
    res = answer("When is the handover of Neptune?", store, llm, k=3)
    assert not res["idk"] and res["lang"] == "en"
    assert res["answer"] == "Handover is December 2027 [1]. Villas are five-bedroom [1]."
    assert len(res["sources"]) == 1 and res["sources"][0]["n"] == 1
    assert "SOURCES:" in llm.last_user and "QUESTION: When is the handover" in llm.last_user


def test_not_found_becomes_i_dont_know_in_the_users_language(store):
    assert answer("Does Dar Global build in Japan?", store, FakeLLM("NOT_FOUND"))["answer"] == IDK["en"]
    res = answer("هل لدى دار جلوبال مشاريع في اليابان؟", store, FakeLLM("NOT_FOUND"))
    assert res["idk"] and res["answer"] == IDK["ar"] and res["sources"] == []


class Down:
    configured = True

    def __init__(self, exc):
        self.exc = exc

    def chat(self, s, u, history=None):
        raise self.exc


def test_llm_outage_degrades_to_sources_not_hallucination(store):
    res = answer("Neptune handover?", store, Down(LLMError("all free models rate-limited")))
    assert res["degraded"] and "temporary outage" in res["answer"] and res["sources"]
    assert "neptune" in res["sources"][0]["url"]
    ar = answer("متى تسليم Neptune؟", store, Down(QuotaExceeded("free-models-per-day")))
    assert ar["degraded"] and "الحد اليومي" in ar["answer"]


def test_follow_up_questions_reuse_the_previous_question(store):
    hist = [{"role": "user", "content": "Tell me about Neptune, Interiors by Mouawad"},
            {"role": "assistant", "content": "Neptune is a villa project in Riyadh [1]."}]
    assert retrieval_query("When is its handover?", hist).startswith("Tell me about Neptune")
    assert retrieval_query("وكم سعرها؟", hist).startswith("Tell me about Neptune")
    assert retrieval_query("Which villas are in Cortesin, Spain on the beach?", hist).startswith("Which villas")
    llm = FakeLLM("December 2027 [1].")
    res = answer("When is its handover?", store, llm, k=3, history=hist)
    assert "neptune" in res["sources"][0]["url"] and llm.history[-1]["role"] == "assistant"


def test_repeated_questions_are_served_from_cache(store):
    llm = FakeLLM("December 2027 [1].")
    first = answer("When is the handover of Neptune?", store, llm, k=3)
    again = answer("  when is the HANDOVER of neptune? ", store, llm, k=3)
    assert llm.calls == 1 and again["cached"] and again["answer"] == first["answer"]
    answer("When is the handover of Neptune?", store, Down(LLMError("x")), k=3)  # cached answer survives outages


def test_echo_llm_end_to_end(store):
    res = answer("Which villas are in Cortesin Spain?", store, EchoLLM())
    assert not res["idk"] and any("marea" in s["url"] for s in res["sources"])


def test_api(store, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "llm", FakeLLM("December 2027 [1]."))
    c = TestClient(main.app)
    r = c.post("/api/chat", json={"question": "Neptune handover?"})
    assert r.status_code == 200 and r.json()["answer"] == "December 2027 [1]."
    r = c.post("/api/chat", json={"question": "and its size?", "history": [
        {"role": "user", "content": "Neptune handover?"}, {"role": "assistant", "content": "December 2027 [1]."}]})
    assert r.status_code == 200
    bad = c.post("/api/chat", json={"question": "hi there", "history": [{"role": "system", "content": "x"}]})
    assert bad.status_code == 422
    assert c.get("/").status_code == 200 and c.get("/api/health").json()["ok"]
    assert c.post("/api/chat", json={"question": ""}).status_code == 422
    for _ in range(int(os.getenv("RATE_LIMIT_PER_MIN", "12"))):
        c.post("/api/chat", json={"question": "Neptune handover?"})
    assert c.post("/api/chat", json={"question": "Neptune handover?"}).status_code == 429


def test_free_model_discovery(monkeypatch):
    import rag.llm as L

    class Resp:
        def json(self):
            return {"data": [{"id": "google/gemma-4-31b-it:free"}, {"id": "acme/new-llama-9b:free"},
                             {"id": "paid/model"}, {"id": "nvidia/nemotron-3-ultra-550b-a55b:free"}]}
    monkeypatch.setattr(L.httpx, "get", lambda *a, **k: Resp())
    m = L.OpenRouterLLM(models="gone/model:free,google/gemma-4-31b-it:free", api_key="x").models
    assert m[0] == "google/gemma-4-31b-it:free" and "gone/model:free" not in m and "paid/model" not in m
    assert "acme/new-llama-9b:free" in m and "nvidia/nemotron-3-ultra-550b-a55b:free" in m
