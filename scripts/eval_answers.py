"""End-to-end answer evaluation against a running deployment (real OpenRouter model).

For each question in eval/questions.jsonl it checks, automatically:
  * unanswerable / off-topic / prompt-injection  -> the bot says "I don't know" (no sources);
  * answerable -> not "I don't know", a cited source matches the expected page, every [n] marker points to a
    listed source, the expected fact (`must` regex) is in the answer, and the answer is in the question's language;
  * partial (e.g. unpublished price) -> no invented price; the answer says the price is not in the data.

    python scripts/eval_answers.py --url https://darglobal-rag-chatbot.onrender.com --ids dg-neptune-handover,na-japan
    python scripts/eval_answers.py --limit 15 --out eval/answer_results.md

The free model tier allows 20 requests/minute and 50/day, so requests are spaced out and the set can be run in parts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.text import detect_lang  # noqa: E402

NO_PRICE = re.compile(r"not (?:published|listed|available|included|provided|in the (?:data|sources))|no (?:unit )?pric|"
                      r"غير (?:منشور|متوفر|متاح|مذكور)|لا (?:تتوفر|يتوفر|توجد|يوجد)", re.I)
PRICE = re.compile(r"\d[\d,.]*\s*(?:million|m\b|SAR|AED|ريال|درهم)", re.I)


def check(case: dict, res: dict) -> tuple[bool, str]:
    ans, srcs = res.get("answer", ""), res.get("sources", [])
    if res.get("degraded"):
        return False, "generation unavailable (degraded)"
    if case.get("unanswerable"):
        return (True, "said I don't know") if res.get("idk") else (False, "answered an unanswerable question")
    if res.get("idk"):
        return False, "said I don't know"
    if case.get("expect") and not any(re.search(case["expect"], s["url"], re.I) for s in srcs):
        return False, "expected page not cited"
    bad = {int(n) for n in re.findall(r"\[(\d+)\]", ans)} - {s["n"] for s in srcs}
    if bad:
        return False, f"dangling citations {sorted(bad)}"
    if case.get("partial"):
        if PRICE.search(ans) and not NO_PRICE.search(ans):
            return False, "stated a price that is not published"
        return (True, "said the price is not in the data") if NO_PRICE.search(ans) else (False, "unclear about price")
    if case.get("must") and not re.search(case["must"], ans, re.I):
        return False, f"missing expected fact /{case['must']}/"
    if detect_lang(ans) != case["lang"]:
        return False, "answered in the wrong language"
    return True, "correct, cited"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://darglobal-rag-chatbot.onrender.com")
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=4.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cases = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.ids:
        wanted = args.ids.split(",")
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]
    rows, ok = [], 0
    for c in cases:
        r = httpx.post(f"{args.url}/api/chat", json={"question": c["q"]}, timeout=120)
        res = r.json() if r.status_code == 200 else {"answer": f"HTTP {r.status_code}"}
        passed, why = check(c, res)
        ok += passed
        rows.append(f"| {c['id']} | {c['lang']} | {'✅' if passed else '❌'} | {why} | {res.get('model') or '-'} | "
                    f"{res.get('latency_ms', '-')} |")
        print(rows[-1], flush=True)
        time.sleep(args.delay)
    md = (f"End-to-end answers - {ok}/{len(cases)} passed ({time.strftime('%Y-%m-%d')}, {args.url})\n\n"
          "| Question | Lang | Pass | Check | Model | ms |\n|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    print("\n" + md.splitlines()[0])
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
