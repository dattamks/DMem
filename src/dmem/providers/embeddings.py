"""Embedding providers.

`EmbeddingProvider` is a tiny protocol: given texts, return vectors. Two
implementations ship:

- `HTTPEmbeddingProvider` — talks to any OpenAI-compatible ``/embeddings``
  endpoint (adopter supplies model + host URL + API key). This covers OpenAI,
  vLLM, TEI, Ollama, LM Studio, and most self-hosted servers.
- `HashingEmbeddingProvider` — a deterministic, dependency-free offline
  embedder. It is NOT semantically strong; it exists purely so the zero-setup
  SQLite tier runs without any network/GPU. The engine warns when it is used.

Choosing a provider is done from config, never auto-magically inside retrieval.
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from typing import Protocol, Sequence, runtime_checkable

from ..config import EmbeddingConfig
from ..errors import ProviderError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        ...

    def signature(self) -> str:
        """Stable identity of this embedder (model + dim). Vectors are only
        comparable across identical signatures."""
        ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class HashingEmbeddingProvider:
    """Deterministic hashing embedder (offline dev/test fallback).

    Uses the hashing trick over word tokens into a fixed-dimensional space,
    then L2-normalizes. Deterministic and dependency-free. Quality is limited
    to lexical overlap — good enough to exercise the full pipeline, and the
    keyword + graph retrieval channels compensate. Adopters should configure a
    real embedding endpoint for production.
    """

    OFFLINE = True

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[idx] += sign
        return _normalize(vec)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._embed_text(text)

    def signature(self) -> str:
        return f"offline-hashing:{self.dim}"


class HTTPEmbeddingProvider:
    """OpenAI-compatible ``/embeddings`` client. Requires the ``http`` extra."""

    OFFLINE = False

    def __init__(self, cfg: EmbeddingConfig):
        if not cfg.host_url:
            raise ProviderError("HTTPEmbeddingProvider requires EMBEDDING_HOST_URL.")
        try:
            import httpx  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on extra
            raise ProviderError(
                "Remote embeddings need the 'http' extra: pip install dmem[http]"
            ) from e
        self._httpx = __import__("httpx")
        self.cfg = cfg
        self.model = cfg.model or "text-embedding-3-small"
        self.dim = cfg.dim or 0  # discovered on first call if unknown
        self._url = cfg.host_url.rstrip("/")
        if not self._url.endswith("/embeddings"):
            self._url = self._url + "/embeddings"

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = self._httpx.post(
                self._url,
                headers=self._headers(),
                json={"model": self.model, "input": list(texts)},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        except Exception as e:  # pragma: no cover - network dependent
            raise ProviderError(f"Embedding request failed: {e}") from e
        vectors = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
        if vectors and not self.dim:
            self.dim = len(vectors[0])
        return [_normalize(v) for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def signature(self) -> str:
        # dim may be 0 until the first call discovers it; include host to
        # disambiguate same model name on different endpoints.
        host = self._url.rsplit("/", 1)[0]
        return f"{self.model}@{host}:{self.dim or '?'}"


def build_embedding_provider(cfg: EmbeddingConfig) -> EmbeddingProvider:
    """Construct the configured embedding provider, or the offline fallback."""
    if cfg.is_remote:
        return HTTPEmbeddingProvider(cfg)
    warnings.warn(
        "DMem: no EMBEDDING_HOST_URL configured — using the offline hashing "
        "embedder. Retrieval quality is limited to lexical overlap. Configure a "
        "real embedding endpoint for production.",
        stacklevel=2,
    )
    return HashingEmbeddingProvider(dim=cfg.dim or 256)
