"""Aggregate documents computed from the Wasalt listings.

RAG retrieves text; it cannot compute statistics over hundreds of listings at question time. So questions like
"what is the cheapest villa in Jeddah?", "average apartment price in Riyadh" or "how much is a 3-bedroom
apartment in Jeddah?" are answered from exact figures computed here at index time.

Each segment (city x property type x purpose) gets several *small, single-purpose* documents - cheapest,
most expensive, price statistics, by number of bedrooms - each opening with the words people use for that
question in English and Arabic. Small focused documents rank far better than one long table, for both BM25
(length normalisation) and the 128-token embedder (it only "sees" the start of a chunk)."""
from __future__ import annotations

import statistics
from collections import defaultdict

CITY = {"Aldammam": ("Dammam", "الدمام"), "Makkah Al Mukarramah": ("Makkah (Mecca)", "مكة"),
        "Riyadh": ("Riyadh", "الرياض"), "Jeddah": ("Jeddah", "جدة")}
TYPE = {"Villa": ("villas", "فلل"), "Apartment": ("apartments", "شقق"), "Land": ("land plots", "أراضي"),
        "All property types": ("properties", "عقارات")}
PURPOSE = {"sale": ("for sale", "للبيع"), "rent": ("for rent", "للإيجار")}


def _fmt(x: float) -> str:
    return f"{int(round(x)):,}"


def _beds(d: dict) -> int | None:
    try:
        return int(str(d["meta"].get("bedrooms") or "").split()[0])
    except (ValueError, IndexError):
        return None


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
        city_en, city_ar = CITY.get(city, (city, ""))
        type_en, type_ar = TYPE.get(subtype, (subtype.lower() + "s", ""))
        purp_en, purp_ar = PURPOSE.get(purpose, (purpose, ""))
        seg_en = f"{type_en} {purp_en} in {city_en}"
        seg_ar = f"{type_ar} {purp_ar} في {city_ar}".strip()
        sample = (f"Computed from a sample of {len(items)} Wasalt listings ({subtype} {purp_en}, {city_en}) collected "
                  f"from Wasalt's public category pages - not the whole market.")
        page = items[0]["meta"].get("category_page") or items[0]["url"]

        def line(d: dict) -> str:
            m, b = d["meta"], _beds(d)
            return (f"{_fmt(m['price'])} {cur}{unit} - {d['title']} - {m.get('district') or city_en} district"
                    f"{f', {b} bedrooms' if b else ''} - {d['url']}")

        def doc(kind: str, title: str, lines: list[str]) -> dict:
            return {"id": f"summary:{city}:{subtype}:{purpose}:{kind}".lower(), "source": "wasalt", "type": "summary",
                    "url": f"{page}#{kind}-{subtype.lower().replace(' ', '-')}", "lang": "en",
                    "title": f"{title} (Wasalt sample)", "text": "\n".join(lines + [sample]),
                    "meta": {"city": city, "count": len(items), "kind": kind}}

        out.append(doc("cheapest", f"Cheapest {seg_en}", [
            f"Cheapest / lowest-priced / least expensive {seg_en} - أرخص {seg_ar}.",
            f"The cheapest is {line(items[0])}.",
            "Next cheapest: " + "; ".join(line(d) for d in items[1:5]) + "."]))
        out.append(doc("most-expensive", f"Most expensive {seg_en}", [
            f"Most expensive / highest-priced {seg_en} - أغلى {seg_ar}.",
            f"The most expensive is {line(items[-1])}.",
            "Next most expensive: " + "; ".join(line(d) for d in reversed(items[-5:-1])) + "."]))
        districts = sorted({d["meta"].get("district") for d in items if d["meta"].get("district")})
        out.append(doc("price-stats", f"Average and median price of {seg_en}", [
            f"Average / median / typical price and price range of {seg_en} - متوسط أسعار {seg_ar}.",
            f"Average {_fmt(statistics.mean(prices))} {cur}{unit}; median {_fmt(statistics.median(prices))} {cur}{unit}; "
            f"range {_fmt(prices[0])} to {_fmt(prices[-1])} {cur}; {len(items)} listings.",
            f"Districts in the sample: {', '.join(districts[:30])}."]))
        by_beds: dict[int, list[float]] = defaultdict(list)
        for d in items:
            if _beds(d):
                by_beds[_beds(d)].append(d["meta"]["price"])
        if by_beds and subtype != "All property types":
            out.append(doc("by-bedrooms", f"Prices of {seg_en} by number of bedrooms", [
                f"Price by number of bedrooms (1, 2, 3, 4, 5+ bedroom) for {seg_en} - الأسعار حسب عدد غرف النوم، {seg_ar}.",
                *(f"{b}-bedroom: {len(p)} listing{'s' if len(p) > 1 else ''}, median {_fmt(statistics.median(p))} {cur}{unit}, "
                  f"from {_fmt(min(p))} to {_fmt(max(p))} {cur}" for b, p in sorted(by_beds.items()))]))
    return out
