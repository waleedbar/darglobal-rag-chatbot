"""Aggregate documents built from Wasalt listings, so questions like "what is the cheapest villa in
Jeddah?" or "average apartment price in Riyadh" can be answered from retrieved text (RAG alone
cannot compute statistics over hundreds of listings)."""
from __future__ import annotations

import statistics
from collections import defaultdict


def _fmt(x: float) -> str:
    return f"{int(round(x)):,}"


def build_summaries(docs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in docs:
        m = d.get("meta") or {}
        if d.get("source") != "wasalt" or d.get("type") != "listing" or not m.get("price") or not m.get("city"):
            continue
        groups[(m["city"], m.get("subtype") or "Property", m.get("purpose") or "sale")].append(d)
        groups[(m["city"], "All property types", m.get("purpose") or "sale")].append(d)
    out = []
    for (city, subtype, purpose), items in sorted(groups.items()):
        if len(items) < 2:
            continue
        items.sort(key=lambda d: d["meta"]["price"])
        prices = [d["meta"]["price"] for d in items]
        cur = items[0]["meta"].get("currency", "SAR")
        unit = " (rent as listed; period not stated)" if purpose == "rent" else ""
        lo, hi = items[0], items[-1]
        districts = sorted({d["meta"].get("district") for d in items if d["meta"].get("district")})
        text = (f"Summary of {len(items)} Wasalt listings: {subtype} for {purpose} in {city}, Saudi Arabia "
                f"(sample collected from Wasalt's public category pages, not the whole market).\n"
                f"Lowest price: {_fmt(prices[0])} {cur}{unit} - {lo['title']} ({lo['meta'].get('district') or city}) - {lo['url']}\n"
                f"Highest price: {_fmt(prices[-1])} {cur}{unit} - {hi['title']} ({hi['meta'].get('district') or city}) - {hi['url']}\n"
                f"Median price: {_fmt(statistics.median(prices))} {cur}{unit}; average: {_fmt(statistics.mean(prices))} {cur}{unit}.\n"
                f"Districts in the sample: {', '.join(districts[:25])}.")
        page = items[0]["meta"].get("category_page") or items[0]["url"]
        out.append({"id": f"summary:{city}:{subtype}:{purpose}".lower(), "source": "wasalt", "type": "summary",
                    "url": page, "title": f"{subtype} for {purpose} in {city} - price summary", "lang": "en",
                    "text": text, "meta": {"city": city, "count": len(items)}})
    return out
