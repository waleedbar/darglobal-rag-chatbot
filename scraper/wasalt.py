"""Wasalt (wasalt.sa): public city/category result pages.

Each category page embeds its listings as JSON (`__NEXT_DATA__`). We keep ONLY an allowlist of
property facts (type, city, district, price, size, rooms, amenities, dates). Owner/agent/contact
fields, coordinates, street names, images and reference numbers are never stored.
`/search` is disallowed by robots.txt, so it is never requested.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from .clean import next_data, scrub
from .http import PoliteClient, RobotsDisallowed

log = logging.getLogger("scraper.wasalt")
BASE = "https://wasalt.sa"

CATEGORIES = [
    # (path, purpose, label)
    ("/en/properties-for-sale-in-riyadh", "sale", "Properties for sale in Riyadh"),
    ("/en/villas-for-sale-in-riyadh", "sale", "Villas for sale in Riyadh"),
    ("/en/apartments-for-sale-in-riyadh", "sale", "Apartments for sale in Riyadh"),
    ("/en/lands-for-sale-in-riyadh", "sale", "Land for sale in Riyadh"),
    ("/en/properties-for-sale-in-jeddah", "sale", "Properties for sale in Jeddah"),
    ("/en/villas-for-sale-in-jeddah", "sale", "Villas for sale in Jeddah"),
    ("/en/apartments-for-sale-in-jeddah", "sale", "Apartments for sale in Jeddah"),
    ("/en/properties-for-sale-in-dammam", "sale", "Properties for sale in Dammam"),
    ("/en/properties-for-sale-in-al-khobar", "sale", "Properties for sale in Al Khobar"),
    ("/en/properties-for-sale-in-makkah", "sale", "Properties for sale in Makkah"),
    ("/en/properties-for-sale-in-madinah", "sale", "Properties for sale in Madinah"),
    ("/en/properties-for-rent-in-riyadh", "rent", "Properties for rent in Riyadh"),
    ("/en/properties-for-rent-in-jeddah", "rent", "Properties for rent in Jeddah"),
    ("/en/properties-for-rent-in-dammam", "rent", "Properties for rent in Dammam"),
]

INFO_FIELDS = ["title", "propertySubType", "propertyMainType", "propertyFor", "city", "state", "territory", "zone",
               "currencyType", "salePrice", "expectedRent", "unitType"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _name(v):
    if isinstance(v, dict):
        return v.get("name") or v.get("title") or v.get("value") or ""
    return v


def parse_listing(p: dict, purpose: str) -> dict | None:
    info = p.get("propertyInfo") or {}
    slug = info.get("slug")
    if not slug:
        return None
    keep = {k: _name(info.get(k)) for k in INFO_FIELDS if info.get(k) not in (None, "", {}, [])}
    attrs = {}
    for a in p.get("attributes") or []:
        if isinstance(a, dict) and a.get("name") and a.get("value") not in (None, ""):
            attrs[a["name"]] = f"{a['value']} {a.get('unit') or ''}".strip()
    amenities = [str(_name(x)) for x in (p.get("featureAmenities") or []) if _name(x)][:25]
    purpose = keep.get("propertyFor") or purpose
    price = keep.get("salePrice") if purpose == "sale" else (keep.get("expectedRent") or keep.get("salePrice"))
    currency = keep.get("currencyType", "SAR")
    district = ", ".join(x for x in [keep.get("zone"), keep.get("territory"), keep.get("city")] if x)
    lines = [f"Listing: {keep.get('title', 'Property')}",
             f"Purpose: for {purpose}",
             f"Type: {keep.get('propertySubType', '')} ({keep.get('propertyMainType', '')})",
             f"Location: {district}, Saudi Arabia",
             (f"{'Rent' if purpose == 'rent' else 'Price'}: {int(float(price)):,} {currency}"
              + (" (rental price as listed on Wasalt; rent period not stated)" if purpose == "rent" else "")) if price else "Price: not listed"]
    lines += [f"{k}: {v}" for k, v in attrs.items()]
    if amenities:
        lines.append("Amenities: " + ", ".join(amenities))
    if p.get("isVerified") or p.get("isRegaProp"):
        lines.append("Verified listing (REGA)")
    if p.get("publishedAt"):
        lines.append(f"Published: {str(p['publishedAt'])[:10]}")
    url = f"{BASE}/en/property/{purpose}/{slug}"
    return {"id": f"wasalt:{p.get('id') or slug}", "source": "wasalt", "type": "listing", "url": url,
            "title": keep.get("title", "Property listing"), "lang": "en", "text": scrub("\n".join(lines)),
            "meta": {"purpose": purpose, "city": keep.get("city"), "district": keep.get("zone"),
                     "subtype": keep.get("propertySubType"), "price": float(price) if price else None,
                     "currency": currency, **{k.lower().replace(" ", "_"): v for k, v in attrs.items()}},
            "scraped_at": _now()}


def scrape_category(client: PoliteClient, path: str, purpose: str, label: str) -> list[dict]:
    url = BASE + path
    r = client.get(url)
    if r.status_code != 200:
        log.warning("%s -> %s", url, r.status_code)
        return []
    data = next_data(r.text)
    if not data:
        log.warning("%s: no __NEXT_DATA__", url)
        return []
    pp = data.get("props", {}).get("pageProps", {})
    props = (pp.get("searchResult") or {}).get("properties") or []
    docs = [d for d in (parse_listing(p, purpose) for p in props) if d]
    for d in docs:
        d["meta"]["category_page"] = url
    # public FAQ shown on the category page (general market Q&A, no personal data)
    faq = pp.get("FAQSchema") or {}
    qa = faq.get("mainEntity") if isinstance(faq, dict) else None
    if isinstance(qa, list) and qa:
        text = "\n".join(f"Q: {q.get('name', '')}\nA: {(q.get('acceptedAnswer') or {}).get('text', '')}"
                         for q in qa if isinstance(q, dict))
        docs.append({"id": f"wasalt:faq:{path}", "source": "wasalt", "type": "page", "url": url,
                     "title": f"{label} - FAQ", "lang": "en", "text": scrub(text)[:6000],
                     "meta": {"category": label}, "scraped_at": _now()})
    log.info("wasalt %s: %d listings", path, len(props))
    return docs


def scrape(client: PoliteClient) -> list[dict]:
    docs: list[dict] = []
    for path, purpose, label in CATEGORIES:
        try:
            docs += scrape_category(client, path, purpose, label)
        except RobotsDisallowed:
            log.info("robots.txt disallows %s - skipped", path)
        except Exception as e:  # noqa: BLE001
            log.warning("failed %s: %s", path, e)
    uniq = {d["id"]: d for d in docs}  # the same listing appears in several categories
    return list(uniq.values())
