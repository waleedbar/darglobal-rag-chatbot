"""Polite HTTP client: obeys robots.txt, rate-limits per host, retries with backoff."""
from __future__ import annotations

import logging
import random
import time
import urllib.robotparser
from urllib.parse import urlparse

import httpx

log = logging.getLogger("scraper.http")

USER_AGENT = ("DarGlobalAssignmentBot/1.0 (+https://github.com/waleedbar/darglobal-rag-chatbot; "
              "educational project, public listing data only)")


class RobotsDisallowed(Exception):
    pass


class PoliteClient:
    def __init__(self, min_delay: float = 2.0, jitter: float = 1.0, timeout: float = 25.0, max_retries: int = 3):
        self.min_delay, self.jitter, self.max_retries = min_delay, jitter, max_retries
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last: dict[str, float] = {}
        self.stats = {"requests": 0, "skipped_by_robots": 0, "errors": 0}
        self.client = httpx.Client(timeout=timeout, follow_redirects=True,
                                   headers={"User-Agent": USER_AGENT, "Accept-Language": "en,ar;q=0.8"})

    # ---- robots.txt -------------------------------------------------------
    def _robots_for(self, url: str) -> urllib.robotparser.RobotFileParser:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self._get_raw(f"{base}/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])  # unreachable robots.txt -> treat as allow-all (standard behaviour)
            self._robots[base] = rp
        return self._robots[base]

    def allowed(self, url: str) -> bool:
        return self._robots_for(url).can_fetch(USER_AGENT, url)

    # ---- fetching ---------------------------------------------------------
    def _wait(self, host: str) -> None:
        delay = self.min_delay + random.uniform(0, self.jitter)
        elapsed = time.monotonic() - self._last.get(host, 0)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last[host] = time.monotonic()

    def _get_raw(self, url: str) -> httpx.Response:
        self._wait(urlparse(url).netloc)
        self.stats["requests"] += 1
        return self.client.get(url)

    def get(self, url: str) -> httpx.Response:
        if not self.allowed(url):
            self.stats["skipped_by_robots"] += 1
            raise RobotsDisallowed(url)
        for attempt in range(self.max_retries + 1):
            try:
                r = self._get_raw(url)
            except httpx.HTTPError as e:
                log.warning("GET %s failed (%s), attempt %d", url, e, attempt + 1)
                time.sleep(2 ** attempt * 2)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After", 2 ** attempt * 3))
                log.warning("GET %s -> %s, backing off %.0fs", url, r.status_code, wait)
                time.sleep(min(wait, 60))
                continue
            return r
        self.stats["errors"] += 1
        raise httpx.HTTPError(f"giving up on {url}")
