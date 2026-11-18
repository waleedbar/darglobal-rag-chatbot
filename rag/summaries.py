"""Aggregate documents computed from the Wasalt listings.

RAG retrieves text; it cannot compute statistics over hundreds of listings at question time. So questions like
"what is the cheapest villa in Jeddah?", "average apartment price in Riyadh" or "how much is a 3-bedroom
apartment in Jeddah?" are answered from exact figures computed here at index time and indexed as ordinary
documents (with links to the listings they mention)."""
from __future__ import annotations

import statistics
from collections import defaultdict

# Wasalt's city labels -> names people actually type (helps keyword retrieval in both languages).
CITY_ALIASES = {"Aldammam": "Dammam (Aldammam, الدمام)", "Makkah Al Mukarramah": "Makkah (Mecca, مكة)",
                "Riyadh": "Riyadh (الرياض)", "Jeddah": "Jeddah (جدة)"}


def _fmt(x: float) -> str:
    return f"{int(round(x)):,}"


def _beds(d: dict) -> int | None:
    try:
        return int(str(d["meta"].get("bedrooms") or "").split()[0])
    except (ValueError, IndexError):
        return None


def _line(d: dict, cur: str, unit: str) -> str:
    m, b = d["meta"], _beds(d)
    beds = f", {b} bedrooms" if b else ""
    return f"{_fmt(m['price'])} {cur}{unit} - {d['title']} ({m.get('district') or m['city']}{beds}) - {d['url']}"


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
        name = CITY_ALIASES.get(city, city)
        districts = sorted({d["meta"].get("district") for d in items if d["meta"].get("district")})
        by_beds: dict[int, list[float]] = defaultdict(list)
        for d in items:
            if _beds(d):
                by_beds[_beds(d)].append(d["meta"]["price"])
        beds_txt = "; ".join(f"{b} bedrooms: {len(p)} listings, median {_fmt(statistics.median(p))} {cur}, "
                             f"range {_fmt(min(p))}-{_fmt(max(p))} {cur}" for b, p in sorted(by_beds.items()))
        text = "\n".join(filter(None, [
            f"Summary of {len(items)} Wasalt listings: {subtype} for {purpose} in {name}, Saudi Arabia "
            f"(a sample collected from Wasalt's public category pages, not the whole market).",
            f"Lowest price: {_line(items[0], cur, unit)}",
            f"Highest price: {_line(items[-1], cur, unit)}",
            f"Median price: {_fmt(statistics.median(prices))} {cur}{unit}; average: {_fmt(statistics.mean(prices))} "
            f"{cur}{unit}.",
            "5 cheapest listings:\n" + "\n".join(f"  {i}. {_line(d, cur, unit)}" for i, d in enumerate(items[:5], 1)),
            "5 most expensive listings:\n" + "\n".join(f"  {i}. {_line(d, cur, unit)}"
                                                      for i, d in enumerate(reversed(items[-5:]), 1)),
            f"By number of bedrooms: {beds_txt}." if beds_txt else "",
            f"Districts in the sample: {', '.join(districts[:30])}.",
        ]))
        page = items[0]["meta"].get("category_page") or items[0]["url"]
        out.append({"id": f"summary:{city}:{subtype}:{purpose}".lower(), "source": "wasalt", "type": "summary",
                    "url": page, "title": f"{subtype} for {purpose} in {city} - price summary", "lang": "en",
                    "text": text, "meta": {"city": city, "count": len(items)}})
    return out
