from __future__ import annotations

import logging
import os
import re

import httpx

log = logging.getLogger("rag.llm")
URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = os.getenv(
    "OPENROUTER_MODELS",
    "google/gemma-4-31b-it:free,nvidia/nemotron-3-ultra-550b-a55b:free,"
    "google/gemma-4-26b-a4b-it:free,baidu/ernie-5.5-flash-preview:free",
)
PREFERRED_FAMILIES = ("gemma", "nemotron", "llama", "qwen", "mistral", "ernie", "deepseek")


class LLMError(RuntimeError):
    pass


class QuotaExceeded(LLMError):
    """OpenRouter's daily cap for free models is account-wide: trying other free models cannot help."""


class OpenRouterLLM:
    def __init__(self, models: str = DEFAULT_MODELS, api_key: str | None = None, timeout: float = 60,
                 discover: bool = True):
        self.models = [m.strip() for m in models.split(",") if m.strip()]
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        if discover:
            self.models = self._live_free_models(self.models)

    @staticmethod
    def _live_free_models(preferred: list[str]) -> list[str]:
        try:
            data = httpx.get("https://openrouter.ai/api/v1/models", timeout=15).json()["data"]
        except Exception as e:  # noqa: BLE001 - offline start: keep the configured list
            log.warning("could not read OpenRouter model list (%s); using configured models", e)
            return preferred
        free = [m["id"] for m in data if m["id"].endswith(":free") and "text" in
                str((m.get("architecture") or {}).get("input_modalities", ["text"]))]
        keep = [m for m in preferred if m in free]
        extra = [m for m in free if m not in keep and any(f in m for f in PREFERRED_FAMILIES)]
        chosen = (keep + extra)[:6] or preferred
        log.info("OpenRouter free models: %s", chosen)
        return chosen

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, history: list[dict] | None = None) -> tuple[str, str]:
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        messages = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}]
        last = ""
        for model in self.models:
            try:
                r = httpx.post(URL, timeout=self.timeout, headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": os.getenv("PUBLIC_URL", "https://github.com/waleedbar/darglobal-rag-chatbot"),
                    "X-Title": "DarGlobal RAG chatbot"},
                    json={"model": model, "temperature": 0.1, "max_tokens": 700,
                          "messages": messages})
                if r.status_code == 200:
                    data = r.json()
                    text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                    if text.strip():
                        return text.strip(), model
                    last = f"{model}: empty response"
                elif r.status_code == 429 and "per-day" in r.text:
                    raise QuotaExceeded(f"daily free-model quota reached ({r.text[:160]})")
                else:
                    last = f"{model}: HTTP {r.status_code} {r.text[:200]}"
            except httpx.HTTPError as e:
                last = f"{model}: {e}"
            log.warning("LLM fallback - %s", last)
        raise LLMError(last or "no model available")


class EchoLLM:
    configured = True

    def chat(self, system: str, user: str, history: list[dict] | None = None) -> tuple[str, str]:
        ctx, _, q = user.partition("QUESTION:")
        words = lambda t: set(re.findall(r"\w{4,}", t.lower()))  # noqa: E731
        qwords = words(q)
        for block in ctx.split("\n[")[1:]:
            n = block.split("]", 1)[0]
            if qwords & words(block):
                return f"According to the source, {block.split(chr(10), 2)[-1][:200].strip()} [{n}]", "echo"
        return "NOT_FOUND", "echo"
