"""Optional LLM provider for high-quality fact extraction & summarization.

Bring-your-own, OpenAI-compatible ``/chat/completions``. When configured, the
typed-fact extractor and the handoff summarizer can use it. When absent, the
engine falls back to the deterministic heuristic extractor and an
extractive summarizer — lower quality but zero-setup and offline.

DMem never bundles or defaults to a specific model.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..config import LLMConfig
from ..errors import ProviderError


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 512, temperature: float = 0.0) -> str:
        ...


class HTTPLLMProvider:
    """OpenAI-compatible chat client. Requires the 'http' extra."""

    def __init__(self, cfg: LLMConfig):
        if not cfg.host_url:
            raise ProviderError("HTTPLLMProvider requires LLM_HOST_URL.")
        try:
            import httpx  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on extra
            raise ProviderError(
                "Remote LLM needs the 'http' extra: pip install dmem[http]"
            ) from e
        self._httpx = __import__("httpx")
        self.cfg = cfg
        self.model = cfg.model or "gpt-4o-mini"
        self._url = cfg.host_url.rstrip("/")
        if not self._url.endswith("/chat/completions"):
            self._url = self._url + "/chat/completions"

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 512, temperature: float = 0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        try:
            resp = self._httpx.post(
                self._url,
                headers=headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:  # pragma: no cover - network dependent
            raise ProviderError(f"LLM request failed: {e}") from e


def build_llm_provider(cfg: LLMConfig) -> Optional[LLMProvider]:
    if cfg.enabled:
        return HTTPLLMProvider(cfg)
    return None
