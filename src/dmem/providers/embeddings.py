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


def _require_httpx(vendor: str):
    try:
        import httpx  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on extra
        raise ProviderError(
            f"{vendor} embeddings need the 'http' extra: pip install dmem[http]"
        ) from e
    return __import__("httpx")


class GoogleEmbeddingProvider:
    """Google Gemini embeddings (``batchEmbedContents``). BYO API key.

    Default endpoint is the public Generative Language API; override host_url for
    a proxy or a Vertex gateway. Request/response shape differs from OpenAI, so
    it is a distinct adapter."""

    OFFLINE = False
    DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, cfg: EmbeddingConfig):
        self._httpx = _require_httpx("Google")
        if not cfg.api_key:
            raise ProviderError("Google embeddings need EMBEDDING_API_KEY.")
        self.cfg = cfg
        self.model = cfg.model or "text-embedding-004"
        self.base = (cfg.host_url or self.DEFAULT_BASE).rstrip("/")
        self.dim = cfg.dim or 0

    def _model_path(self) -> str:
        return self.model if self.model.startswith("models/") else f"models/{self.model}"

    def _endpoint(self) -> str:
        return f"{self.base}/{self._model_path()}:batchEmbedContents?key={self.cfg.api_key}"

    def _payload(self, texts: Sequence[str]) -> dict:
        mp = self._model_path()
        return {"requests": [{"model": mp, "content": {"parts": [{"text": t}]}}
                             for t in texts]}

    def _parse(self, data: dict) -> list[list[float]]:
        return [_normalize(e["values"]) for e in data.get("embeddings", [])]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = self._httpx.post(self._endpoint(), json=self._payload(texts),
                                    timeout=60.0)
            resp.raise_for_status()
            vecs = self._parse(resp.json())
        except Exception as e:  # pragma: no cover - network dependent
            raise ProviderError(f"Google embedding request failed: {e}") from e
        if vecs and not self.dim:
            self.dim = len(vecs[0])
        return vecs

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def signature(self) -> str:
        return f"google:{self.model}:{self.dim or '?'}"


class CohereEmbeddingProvider:
    """Cohere embeddings (v2 ``/embed``). BYO API key.

    Distinct request/response shape from OpenAI. Uses ``search_document`` input
    type for stored content (a v1 simplification — asymmetric query typing is a
    future optimization)."""

    OFFLINE = False
    DEFAULT_BASE = "https://api.cohere.com/v2"

    def __init__(self, cfg: EmbeddingConfig):
        self._httpx = _require_httpx("Cohere")
        if not cfg.api_key:
            raise ProviderError("Cohere embeddings need EMBEDDING_API_KEY.")
        self.cfg = cfg
        self.model = cfg.model or "embed-v4.0"
        base = (cfg.host_url or self.DEFAULT_BASE).rstrip("/")
        self._url = base if base.endswith("/embed") else base + "/embed"
        self.dim = cfg.dim or 0

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json"}

    def _payload(self, texts: Sequence[str],
                 input_type: str = "search_document") -> dict:
        return {"model": self.model, "texts": list(texts),
                "input_type": input_type, "embedding_types": ["float"]}

    def _parse(self, data: dict) -> list[list[float]]:
        embs = data.get("embeddings", {})
        floats = embs.get("float") if isinstance(embs, dict) else embs
        return [_normalize(v) for v in (floats or [])]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = self._httpx.post(self._url, headers=self._headers(),
                                    json=self._payload(texts), timeout=60.0)
            resp.raise_for_status()
            vecs = self._parse(resp.json())
        except Exception as e:  # pragma: no cover - network dependent
            raise ProviderError(f"Cohere embedding request failed: {e}") from e
        if vecs and not self.dim:
            self.dim = len(vecs[0])
        return vecs

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def signature(self) -> str:
        return f"cohere:{self.model}:{self.dim or '?'}"


# Provider aliases -> canonical name. OpenAI-compatible covers the long tail of
# vendors that expose an OpenAI-style /embeddings endpoint.
_OPENAI_COMPATIBLE = {"openai", "openai-compatible", "compatible", "vllm", "tei",
                      "ollama", "lmstudio", "azure", "dashscope", "qwen",
                      "together", "fireworks", "deepinfra", "voyage-openai"}


def build_embedding_provider(cfg: EmbeddingConfig) -> EmbeddingProvider:
    """Construct the configured embedding provider, or the offline fallback.

    Selected by ``EMBEDDING_PROVIDER`` (cfg.provider). Empty => auto:
    OpenAI-compatible when a host_url is set, else the offline dev embedder."""
    provider = (cfg.provider or "").strip().lower()

    if provider in ("google", "gemini"):
        return GoogleEmbeddingProvider(cfg)
    if provider == "cohere":
        return CohereEmbeddingProvider(cfg)
    if provider in ("offline", "hashing", "none", "dev"):
        return HashingEmbeddingProvider(dim=cfg.dim or 256)
    if provider in _OPENAI_COMPATIBLE:
        if not cfg.host_url:
            raise ProviderError(
                f"EMBEDDING_PROVIDER={cfg.provider!r} is OpenAI-compatible and "
                "needs EMBEDDING_HOST_URL (your vendor's base URL).")
        return HTTPEmbeddingProvider(cfg)
    if provider:
        raise ProviderError(
            f"Unknown EMBEDDING_PROVIDER={cfg.provider!r}. Use one of: openai "
            "(and OpenAI-compatible vendors), google, cohere, offline.")

    # auto
    if cfg.host_url:
        return HTTPEmbeddingProvider(cfg)
    warnings.warn(
        "DMem: no EMBEDDING_HOST_URL / EMBEDDING_PROVIDER configured — using the "
        "offline hashing embedder. Retrieval quality is limited to lexical "
        "overlap. Configure a real embedding endpoint for production.",
        stacklevel=2,
    )
    return HashingEmbeddingProvider(dim=cfg.dim or 256)
