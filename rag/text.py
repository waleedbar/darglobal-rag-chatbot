"""Language detection and Arabic-aware tokenisation (used for BM25 and the answer language)."""
from __future__ import annotations

import re

AR = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
STOP = set("""the a an of to in on for and or is are was were be with at by from as this that it its what which who how
much many do does any there their about me i you we can please في من على الى إلى عن ما هل كم هي هو التي الذي او أو و ان أن""".split())


def detect_lang(text: str) -> str:
    return "ar" if len(AR.findall(text)) >= max(1, len(LATIN.findall(text))) * 0.5 else "en"


def normalize(text: str) -> str:
    t = _DIACRITICS.sub("", text.translate(_AR_DIGITS).lower())
    t = re.sub("[إأآا]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه")
    return t


def tokenize(text: str) -> list[str]:
    toks = []
    for w in _TOKEN.findall(normalize(text)):
        if w in STOP or len(w) < 2:
            continue
        if AR.search(w) and w.startswith("ال") and len(w) > 4:  # strip Arabic definite article
            w = w[2:]
        toks.append(w)
    return toks
