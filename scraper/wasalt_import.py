"""Import a Wasalt snapshot collected from a normal browser session, and (de)crypt it.

Why: wasalt.sa answers 403 to automated requests from cloud IPs (GitHub Actions, Hugging Face),
although robots.txt and https://wasalt.sa/llms.txt allow extracting public listing content at a
moderate rate. The snapshot was therefore collected by visiting 8 public category pages in a regular
browser (one page every ~10 s) and extracting only allowlisted listing fields in the page itself
(see docs/wasalt_browser_collector.js). Wasalt's policy forbids redistributing the data, so only an
encrypted copy is committed; the Docker build decrypts it with the DATA_KEY build secret.

    python -m scraper.wasalt_import import  wasalt_snapshot.jsonl   # -> data/raw/wasalt.jsonl
    python -m scraper.wasalt_import encrypt                          # -> data/private/wasalt.jsonl.enc
    python -m scraper.wasalt_import decrypt                          # (Docker build) -> data/raw/wasalt.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from .wasalt import parse_listing
from .clean import scrub

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "wasalt.jsonl"
ENC = ROOT / "data" / "private" / "wasalt.jsonl.enc"


def import_snapshot(path: str) -> int:
    docs: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        page = json.loads(line)
        for p in page.get("properties", []):
            d = parse_listing(p, page.get("purpose", "sale"))
            if d:
                d["meta"]["category_page"] = page["url"]
                d["scraped_at"] = page.get("collected_at", "")
                docs.setdefault(d["id"], d)
        faq = page.get("faq") or []
        if faq:
            text = "\n".join(f"Q: {q.get('name', '')}\nA: {(q.get('acceptedAnswer') or {}).get('text', '')}" for q in faq)
            docs[f"wasalt:faq:{page['url']}"] = {
                "id": f"wasalt:faq:{page['url']}", "source": "wasalt", "type": "page", "url": page["url"],
                "title": f"{page.get('label', 'Wasalt')} - FAQ", "lang": "en", "text": scrub(text)[:6000],
                "meta": {"category": page.get("label")}, "scraped_at": page.get("collected_at", "")}
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs.values()), encoding="utf-8")
    return len(docs)


def _key() -> str:
    """The Fernet key from DATA_KEY, tolerant of how secret stores hand it over
    (trailing newline, a dropped '=' padding, quotes/backticks, a `DATA_KEY=` prefix, BOM or invisible direction marks)."""
    raw = os.environ.get("DATA_KEY", "")
    m = re.search(r"[A-Za-z0-9_-]{43}", raw)
    return m.group(0) + "=" if m else raw.strip()  # the "=" padding is often lost when copy-pasting


def _fernet():
    from cryptography.fernet import Fernet
    key = _key()
    if not key:
        sys.exit("DATA_KEY is not set")
    return Fernet(key.encode())


def encrypt() -> None:
    ENC.parent.mkdir(parents=True, exist_ok=True)
    ENC.write_bytes(_fernet().encrypt(RAW.read_bytes()))


def decrypt() -> None:
    if not ENC.exists():
        print("no encrypted Wasalt snapshot - building with Dar Global data only")
        return
    if not os.environ.get("DATA_KEY"):
        print("DATA_KEY not provided - building with Dar Global data only")
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    try:
        RAW.write_bytes(_fernet().decrypt(ENC.read_bytes()))
    except Exception as exc:  # wrong/garbled key: still ship the Dar Global bot, say why (never print the key)
        raw = os.environ.get("DATA_KEY", "")
        print(f"WARNING: could not decrypt the Wasalt snapshot ({type(exc).__name__}); DATA_KEY has "
              f"{len(raw)} chars, {len(_key())} after cleaning - building with Dar Global data only")
        return
    print(f"decrypted {RAW} ({RAW.stat().st_size} bytes)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "import":
        print(f"wrote {import_snapshot(sys.argv[2])} Wasalt documents to {RAW}")
    elif cmd == "encrypt":
        encrypt()
        print(f"wrote {ENC}")
    elif cmd == "decrypt":
        decrypt()
    else:
        sys.exit(__doc__)
