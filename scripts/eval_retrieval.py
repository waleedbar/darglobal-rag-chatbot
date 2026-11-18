"""Retrieval sanity check: does the right page appear in the top-k for known questions (EN + AR)?

    python scripts/eval_retrieval.py            # uses the built index and the configured embedder
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embeddings import get_embedder  # noqa: E402
from rag.store import Store  # noqa: E402

CASES = [
    ("When is the handover of Neptune, Interiors by Mouawad?", "darglobal.co.uk/neptune"),
    ("How many floors does Trump Tower Jeddah have?", "trump-tower-jeddah"),
    ("Which Dar Global project has interiors by Aston Martin?", "the-astera"),
    ("Tell me about the Lamborghini-inspired villas in Spain", "tierra-viva|painite"),
    ("Who is the founder and CEO of Dar Global?", "darglobal.co.uk/about"),
    ("What is the cheapest villa for sale in Jeddah?", "villas-for-sale-in-jeddah|wasalt.sa/en/property/sale/villa"),
    ("apartments for rent in Riyadh price", "apartments-for-rent-in-riyadh|property/rent"),
    ("متى موعد تسليم مشروع نبتون في الرياض؟", "darglobal.co.uk/neptune"),
    ("ما هي مشاريع دار جلوبال في عمان؟", "aida|oman"),
    ("كم عدد طوابق برج ترامب جدة؟", "trump-tower-jeddah"),
    ("فلل للبيع في جدة", "villa|jeddah"),
    ("مشروع في لندن من دار جلوبال", "mulliner|albert-hall"),
]


def main(k: int = 5) -> None:
    import re
    store = Store(get_embedder())
    ok = 0
    for q, expect in CASES:
        hits = store.search(q, k=k)
        found = any(re.search(expect, h["url"], re.I) for h in hits)
        ok += found
        print(f"{'PASS' if found else 'MISS'}  {q}")
        for h in hits[:3]:
            print(f"        {h['score']:.3f}  {h['url']}")
    print(f"\nhit@{k}: {ok}/{len(CASES)}")


if __name__ == "__main__":
    main()
