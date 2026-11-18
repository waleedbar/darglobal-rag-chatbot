"""Dar Global (darglobal.co.uk): project catalogue, project pages and developer pages."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .clean import next_data, scrub, visible_text
from .http import PoliteClient, RobotsDisallowed

log = logging.getLogger("scraper.darglobal")
BASE = "https://darglobal.co.uk"
LISTING = f"{BASE}/projects"
SEEDS = Path(__file__).with_name("seeds") / "darglobal_projects.txt"
NAV = re.compile(r"^/(|projects/?|blog.*|press.*|contact.*|careers.*|privacy.*|terms.*|cookie.*|investor.*|"
                 r"media.*|news.*|ar/?|en/?|search.*|register.*)$", re.I)
DEV_PAGE = re.compile(r"/(about|who-we-are|our-story|leadership|management|board|company|dar-global)", re.I)

PRICE = re.compile(r"(?:(AED|SAR|GBP|EUR|USD|OMR|QAR|£|€|\$)\s?([\d,.]+\s?(?:M|K|million|bn)?)"
                   r"|([\d,.]+\s?(?:M|K|million)?)\s?(AED|SAR|GBP|EUR|USD|OMR|QAR))", re.I)
BEDS = re.compile(r"(\d)\s*(?:-|to|&|and)?\s*(\d)?\s*(?:bed|br\b|bedroom)", re.I)
HANDOVER = re.compile(r"(?:handover|completion|delivery)[^.\n]{0,40}?((?:Q[1-4]\s*)?(?:20\d\d)|"
                      r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+20\d\d)", re.I)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def discover_projects(client: PoliteClient) -> dict[str, str]:
    """Return {url: 'Name | Location'} from the public /projects page, merged with the seed list."""
    found: dict[str, str] = {}
    for line in SEEDS.read_text(encoding="utf-8").splitlines():  # seeds first: they carry "Name | Location"
        if line.strip() and not line.startswith("#"):
            url, _, label = line.partition("|")
            found[url.strip().rstrip("/")] = label.strip()
    try:
        html = client.get(LISTING).text
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = urljoin(BASE, a["href"].split("?")[0].split("#")[0])
            path = urlparse(href).path
            if urlparse(href).netloc.replace("www.", "") != "darglobal.co.uk" or NAV.match(path):
                continue
            text = " ".join(a.get_text(" ", strip=True).split())
            if a.find("img") or ("," in text and len(text) < 160):
                href = href.replace("://www.", "://").rstrip("/")
                if href.lower() not in {u.lower() for u in found}:  # paths are case-sensitive; dedupe ignoring case
                    found[href] = text
    except Exception as e:  # noqa: BLE001 - listing page is a convenience, seeds still work
        log.warning("could not read listing page: %s", e)
    return found


def embedded_text(html: str, max_chars: int = 12000) -> str:
    """Collect human-readable strings from __NEXT_DATA__ (titles, descriptions, specs), skipping URLs/ids/code."""
    data = next_data(html)
    if not data:
        return ""
    out, seen = [], set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            t = re.sub(r"<[^>]+>", " ", o)
            t = " ".join(t.split())
            if (len(t) >= 12 and " " in t and t not in seen and not re.match(r"^(https?:|/|#|\{|data:)", t)
                    and not re.search(r"\.(jpg|png|webp|svg|mp4)\b", t, re.I)):
                seen.add(t)
                out.append(t)

    walk(data.get("props", {}).get("pageProps", {}))
    return scrub("\n".join(out))[:max_chars]


def facts(text: str) -> dict:
    prices = list(dict.fromkeys(" ".join(filter(None, m)).strip() for m in PRICE.findall(text)))[:5]
    beds = sorted({int(x) for m in BEDS.findall(text) for x in m if x})
    hand = HANDOVER.search(text)
    return {"prices_mentioned": prices, "bedrooms": beds,
            "handover": hand.group(1) if hand else None,
            "prices_published": bool(prices)}


def scrape_project(client: PoliteClient, url: str, label: str) -> dict | None:
    r = client.get(url)
    if r.status_code != 200:
        log.warning("%s -> %s", url, r.status_code)
        return None
    title, desc, text = visible_text(r.text)
    if len(text) < 400:  # client-rendered page: fall back to the text embedded in Next.js page data
        text = (text + "\n" + embedded_text(r.text)).strip()
    name, _, location = label.partition("|") if "|" in label else (label, "", "")
    f = facts(text)
    header = [f"Project: {title or name.strip()}", "Developer: Dar Global (DarGlobal PLC)"]
    if location.strip():
        header.append(f"Location: {location.strip()}")
    if desc:
        header.append(f"Summary: {scrub(desc)}")
    if f["handover"]:
        header.append(f"Handover/completion: {f['handover']}")
    header.append("Prices on page: " + (", ".join(f["prices_mentioned"]) if f["prices_published"]
                                        else "not published on the project page (contact sales)"))
    return {"id": "dg:" + urlparse(url).path.strip("/").lower(), "source": "darglobal", "type": "project",
            "url": url, "title": title or name.strip(), "lang": "en",
            "text": "\n".join(header) + "\n\n" + text, "meta": {"location": location.strip(), **f},
            "scraped_at": _now()}


def scrape_developer_pages(client: PoliteClient, limit: int = 4) -> list[dict]:
    docs = []
    try:
        xml = client.get(f"{BASE}/sitemap.xml").text
        urls = [u for u in re.findall(r"<loc>(.*?)</loc>", xml) if DEV_PAGE.search(urlparse(u).path)]
    except Exception as e:  # noqa: BLE001
        log.warning("sitemap unavailable: %s", e)
        urls = []
    for u in (urls or [f"{BASE}/about-us"])[:limit]:
        try:
            r = client.get(u)
        except RobotsDisallowed:
            continue
        if r.status_code != 200:
            continue
        title, desc, text = visible_text(r.text)
        docs.append({"id": "dg:" + urlparse(u).path.strip("/").lower(), "source": "darglobal", "type": "developer",
                     "url": u, "title": title or "About Dar Global", "lang": "en",
                     "text": f"Developer: Dar Global\n{scrub(desc)}\n\n{text}", "meta": {}, "scraped_at": _now()})
    return docs


def scrape(client: PoliteClient, max_projects: int = 60) -> list[dict]:
    docs = scrape_developer_pages(client)
    projects = discover_projects(client)
    log.info("darglobal: %d project URLs", len(projects))
    for url, label in list(projects.items())[:max_projects]:
        try:
            d = scrape_project(client, url, label)
            if d and len(d["text"]) > 200:
                docs.append(d)
        except RobotsDisallowed:
            log.info("robots.txt disallows %s - skipped", url)
        except Exception as e:  # noqa: BLE001
            log.warning("failed %s: %s", url, e)
    return docs
