"""Split documents into retrieval chunks. Every chunk carries a short header (title, source, location)
so it stays meaningful on its own ("contextual chunk headers")."""
from __future__ import annotations

import re

MAX_CHARS = 1000
OVERLAP = 150


def _header(doc: dict) -> str:
    src = {"darglobal": "Dar Global", "wasalt": "Wasalt"}.get(doc["source"], doc["source"])
    loc = (doc.get("meta") or {}).get("location") or ""
    return f"[{src} | {doc.get('type', '')} | {doc['title']}{' | ' + loc if loc else ''}]"


def split_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n(?=## )|\n{2,}", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > max_chars:  # very long paragraph: hard split on sentence/line boundaries
            cut = max(p.rfind(". ", 0, max_chars), p.rfind("\n", 0, max_chars))
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:cut].strip())
            p = p[cut - overlap if cut > overlap else cut:].strip()
        if len(cur) + len(p) + 1 > max_chars and cur:
            chunks.append(cur)
            cur = cur[-overlap:].split("\n", 1)[-1] + "\n" + p if overlap else p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def chunk_doc(doc: dict) -> list[dict]:
    head = _header(doc)
    out = []
    for i, body in enumerate(split_text(doc["text"])):
        out.append({"chunk_id": f"{doc['id']}#{i}", "doc_id": doc["id"], "source": doc["source"],
                    "type": doc.get("type"), "title": doc["title"], "url": doc["url"],
                    "text": f"{head}\n{body}"})
    return out
