from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup

PHONE = re.compile(r"(?<![\w/,.])\+?\d[\d\s\-()]{7,}\d(?![\w/,])")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
BOILERPLATE = re.compile(r"(cookie|privacy policy|terms (and|&) conditions|all rights reserved|subscribe to|"
                         r"newsletter|accept all|whatsapp|call us|download (the )?app|©)", re.I)


def scrub(text: str) -> str:
    text = EMAIL.sub("[removed]", text)
    return PHONE.sub(lambda m: "[removed]" if len(re.sub(r"\D", "", m.group())) >= 9 else m.group(), text)


def visible_text(html: str, max_chars: int = 15000) -> tuple[str, str, str]:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.find("meta", property="og:title") or {}).get("content") or (soup.title.string if soup.title else "")
    desc = (soup.find("meta", attrs={"name": "description"}) or {}).get("content", "")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form", "iframe"]):
        tag.decompose()
    root = soup.find("main") or soup.body or soup
    lines, seen = [], set()
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "dt", "dd", "span", "div"]):
        if el.find(["p", "li", "div", "h1", "h2", "h3", "table"]):
            continue
        t = " ".join(el.get_text(" ", strip=True).split())
        if len(t) < 3 or t in seen or BOILERPLATE.search(t):
            continue
        if lines and (t in lines[-1] or (len(lines) > 1 and t in lines[-2])):
            continue
        seen.add(t)
        lines.append(f"## {t}" if el.name in ("h1", "h2", "h3") else t)
    text = scrub("\n".join(lines))[:max_chars]
    h1 = soup.find("h1")
    return (h1.get_text(" ", strip=True) if h1 else (title or "")).strip(), desc.strip(), text


def next_data(html: str) -> dict | None:
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
