"""Retrieval evaluation on the labelled EN/AR question set (eval/questions.jsonl), with an ablation.

For every answerable question we know which page(s) hold the answer (`expect`, a regex on the URL).
Reports hit@1, hit@5 and MRR@10 per language for hybrid (dense + BM25, RRF) vs dense-only vs BM25-only,
optionally for several chunk sizes (each one rebuilds a temporary index):

    python scripts/eval_retrieval.py                       # built index, all three modes
    python scripts/eval_retrieval.py --chunks 500,1000     # also compare chunk sizes
    python scripts/eval_retrieval.py --out eval/retrieval_results.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.embeddings import get_embedder  # noqa: E402
from rag.store import INDEX_DIR, Store, build  # noqa: E402

MODES = ("hybrid", "dense", "bm25")


def load_cases() -> list[dict]:
    rows = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [r for r in rows if r.get("expect")]


def rank_of(hits: list[dict], pattern: str) -> int | None:
    for i, h in enumerate(hits, 1):
        if re.search(pattern, h["url"], re.I):
            return i
    return None


def evaluate(store: Store, cases: list[dict], mode: str) -> dict:
    per = {"all": [], "en": [], "ar": []}
    misses = []
    for c in cases:
        r = rank_of(store.search(c["q"], k=10, mode=mode), c["expect"])
        for key in ("all", c["lang"]):
            per[key].append(r)
        if not r or r > 5:
            misses.append(c["id"])
    out = {}
    for key, ranks in per.items():
        n = len(ranks) or 1
        out[key] = {"n": len(ranks), "hit@1": sum(1 for r in ranks if r == 1) / n,
                    "hit@5": sum(1 for r in ranks if r and r <= 5) / n,
                    "mrr@10": sum(1 / r for r in ranks if r) / n}
    out["misses@5"] = misses
    return out


def table(results: dict) -> str:
    lines = ["| Index | Retriever | hit@1 | hit@5 | MRR@10 | hit@5 EN | hit@5 AR |", "|---|---|---|---|---|---|---|"]
    for (label, mode), r in results.items():
        a = r["all"]
        lines.append(f"| {label} | {mode} | {a['hit@1']:.2f} | {a['hit@5']:.2f} | {a['mrr@10']:.2f} | "
                     f"{r['en']['hit@5']:.2f} | {r['ar']['hit@5']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="", help="comma-separated chunk sizes to compare (rebuilds temp indexes)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    emb, cases = get_embedder(), load_cases()
    stores = [("built index", Store(emb, INDEX_DIR))]
    for size in [int(s) for s in args.chunks.split(",") if s.strip()]:
        tmp = Path(tempfile.mkdtemp())
        build(emb, out=tmp, chunk_chars=size)
        stores.append((f"{size}-char chunks", Store(emb, tmp)))
    results = {}
    for label, store in stores:
        for mode in MODES:
            results[(label, mode)] = evaluate(store, cases, mode)
    md = (f"Retrieval evaluation - {len(cases)} labelled questions "
          f"({sum(c['lang'] == 'en' for c in cases)} EN / {sum(c['lang'] == 'ar' for c in cases)} AR), "
          f"embedder `{emb.name}`, chunks {stores[0][1].stats['chunks']}.\n\n" + table(results))
    misses = results[("built index", "hybrid")]["misses@5"]
    md += f"\n\nHybrid misses (not in top 5): {', '.join(misses) or 'none'}\n"
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
