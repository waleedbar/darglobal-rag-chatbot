"""Run the scrapers and write data/raw/<source>.jsonl.

    python -m scraper.run                 # both sites
    python -m scraper.run --only wasalt   # one site
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import darglobal, wasalt
from .http import PoliteClient

OUT = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["darglobal", "wasalt"])
    ap.add_argument("--delay", type=float, default=2.0, help="minimum seconds between requests to the same host")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    client = PoliteClient(min_delay=args.delay)
    for name, mod in (("darglobal", darglobal), ("wasalt", wasalt)):
        if args.only and args.only != name:
            continue
        docs = mod.scrape(client)
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        logging.info("%s: wrote %d documents", name, len(docs))
    logging.info("http stats: %s", client.stats)


if __name__ == "__main__":
    main()
