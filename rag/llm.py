"""Generation via OpenRouter (free models), with a fallback chain across several free models.

The API key comes only from the OPENROUTER_API_KEY environment variable (a platform secret)."""
from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger("rag.llm")
URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = os.getenv(
    "OPENROUTER_MODELS",
    "meta-llama/llama-3.3-70b-instruct:free,google/gemma-3-27b-it:free,"
    "mistralai/mistral-small-3.2-24b-instruct:free,qwen/qwen3-235b-a22b:free",
)


class LLMError(RuntimeError):
    pass


class OpenRouterLLM:
    def __init__(self, models: str = DEFAULT_MODELS, api_key: str | None = None, timeout: float = 60):
        self.models = [m.strip() for m in models.split(",") if m.strip()]
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str) -> tuple[str, str]:
        """Return (text, model_used). Tries each free model in order on rate limits / outages."""
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        last = ""
        for model in self.models:
            try:
                r = httpx.post(URL, timeout=self.timeout, headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": os.getenv("PUBLIC_URL", "https://github.com/waleedbar/darglobal-rag-chatbot"),
                    "X-Title": "DarGlobal RAG chatbot"},
                    json={"model": model, "temperature": 0.1, "max_tokens": 700,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
                if r.status_code == 200:
                    data = r.json()
                    text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                    if text.strip():
                        return text.strip(), model
                    last = f"{model}: empty response"
                else:
                    last = f"{model}: HTTP {r.status_code} {r.text[:200]}"
            except httpx.HTTPError as e:
                last = f"{model}: {e}"
            log.warning("LLM fallback - %s", last)
        raise LLMError(last or "no model available")


class EchoLLM:
    """Offline stand-in for tests: answers with the first context line that shares words with the question."""

    configured = True

    def chat(self, system: str, user: str) -> tuple[str, str]:
        ctx, _, q = user.partition("QUESTION:")
        qwords = {w for w in q.lower().split() if len(w) > 3}
        for block in ctx.split("\n[")[1:]:
            n = block.split("]", 1)[0]
            if qwords & set(block.lower().split()):
                return f"According to the source, {block.split(chr(10), 2)[-1][:200].strip()} [{n}]", "echo"
        return "NOT_FOUND", "echo"
