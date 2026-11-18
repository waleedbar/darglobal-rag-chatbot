"""RAG pipeline tests with a fixed mini-corpus, the hashing embedder and fake LLMs (fully offline)."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from rag.chunking import split_text
from rag.embeddings import HashEmbedder
from rag.llm import EchoLLM, LLMError
from rag.pipeline import IDK, UNAVAILABLE, answer
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
    assert stats["docs"] == len(DOCS) + 2  # + 2 price summaries (Villa / All types in Jeddah)
    return Store(HashEmbedder(), index_dir=tmp / "index")


class FakeLLM:
    configured = True

    def __init__(self, reply):
        self.reply, self.last_user = reply, ""

    def chat(self, system, user):
        self.last_user = user
        return self.reply, "fake/model:free"


def test_language_detection_and_arabic_tokens():
    assert detect_lang("متى موعد تسليم مشروع نبتون؟") == "ar"
    assert detect_lang("When is the Neptune handover?") == "en"
    assert detect_lang("كم سعر DG1؟") == "ar"
    assert "مشروع" in tokenize("المشروع") or "مشروع" in tokenize("مشروع")


def test_chunking_respects_size_and_keeps_text():
    text = "\n\n".join(f"## Section {i}\n" + "word " * 120 for i in range(6))
    chunks = split_text(text)
    assert all(len(c) <= 1100 for c in chunks) and len(chunks) >= 3
    assert "Section 5" in chunks[-1]


def test_hybrid_retrieval_finds_the_right_project(store):
    hits = store.search("When is the handover of Neptune?", k=3)
    assert hits[0]["url"] == "https://darglobal.co.uk/neptune"


def test_price_summary_documents():
    s = {d["title"]: d for d in build_summaries(DOCS)}
    t = s["Villa for sale in Jeddah - price summary"]["text"]
    assert "Lowest price: 1,200,000 SAR" in t and "Highest price: 3,500,000 SAR" in t and "Median price: 2,000,000" in t


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


def test_llm_outage_is_reported_not_hallucinated(store):
    class Down:
        configured = True

        def chat(self, s, u):
            raise LLMError("all free models rate-limited")
    res = answer("Neptune handover?", store, Down())
    assert res["answer"] == UNAVAILABLE["en"] and res.get("error")


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
    assert c.get("/").status_code == 200 and c.get("/api/health").json()["ok"]
    assert c.post("/api/chat", json={"question": ""}).status_code == 422
    for _ in range(int(os.getenv("RATE_LIMIT_PER_MIN", "12"))):
        c.post("/api/chat", json={"question": "Neptune handover?"})
    assert c.post("/api/chat", json={"question": "Neptune handover?"}).status_code == 429
