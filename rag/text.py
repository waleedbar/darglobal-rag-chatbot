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


# Arabic place names -> English forms used in the (mostly English) source pages. Helps both BM25 and the
# dense retriever, and disambiguates "عمان" (Oman vs. Amman) towards Oman, where Dar Global builds.
PLACES = {"الرياض": "Riyadh", "جدة": "Jeddah", "جده": "Jeddah", "مكة": "Makkah", "مكه": "Makkah",
          "المدينة": "Madinah", "الدمام": "Dammam", "الخبر": "Khobar", "دبي": "Dubai", "أبوظبي": "Abu Dhabi",
          "رأس الخيمة": "Ras Al Khaimah RAK", "عمان": "Oman Muscat", "عُمان": "Oman Muscat", "مسقط": "Muscat Oman",
          "قطر": "Qatar Doha", "الدوحة": "Doha Qatar", "لندن": "London UK", "إسبانيا": "Spain", "اسبانيا": "Spain",
          "السعودية": "Saudi Arabia KSA", "الإمارات": "UAE", "الامارات": "UAE", "ترامب": "Trump", "فلل": "villas",
          "فيلا": "villa", "شقق": "apartments", "شقة": "apartment", "أرض": "land", "اراضي": "land", "إيجار": "rent",
          "ايجار": "rent", "للبيع": "for sale", "تسليم": "handover", "سعر": "price", "أسعار": "prices"}


def expand_query(q: str) -> str:
    extra = [en for ar, en in PLACES.items() if ar in q]
    return q + (" (" + " ".join(dict.fromkeys(extra)) + ")" if extra else "")


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
