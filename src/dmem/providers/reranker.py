"""Cross-encoder reranking.

The brief requires a *lightweight cross-encoder* rerank step — explicitly NOT an
LLM call, to avoid re-introducing the latency/cost the system exists to avoid.

- `CrossEncoderReranker` uses ``sentence-transformers`` (the ``rerank`` extra)
  with an adopter-supplied model name.
- `LexicalReranker` is a dependency-free fallback that scores query/candidate
  lexical overlap. It keeps the pipeline honest (a real rerank *ordering* step
  exists at every tier) without requiring a model download.

Core degrades gracefully: if reranking is disabled or unavailable, retrieval
returns fusion-scored results unchanged.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Protocol, Sequence, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]:
        """Return a relevance score per candidate (higher = more relevant)."""
        ...


class LexicalReranker:
    """Dependency-free fallback reranker based on token overlap + IDF-ish weighting."""

    OFFLINE = True

    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]:
        q = set(_TOKEN_RE.findall(query.lower()))
        if not q:
            return [0.0 for _ in candidates]
        scores: list[float] = []
        for c in candidates:
            ct = _TOKEN_RE.findall(c.lower())
            if not ct:
                scores.append(0.0)
                continue
            cset = set(ct)
            overlap = len(q & cset)
            # Jaccard-ish, dampened by length so long chunks don't dominate.
            score = overlap / (math.sqrt(len(cset)) + 1.0)
            scores.append(score)
        return scores


class CrossEncoderReranker:
    """Real cross-encoder via sentence-transformers. Requires the 'rerank' extra."""

    OFFLINE = False

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover - depends on extra
            raise ImportError(
                "Cross-encoder reranking needs the 'rerank' extra: "
                "pip install dmem[rerank]"
            ) from e
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, c) for c in candidates]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


def build_reranker(
    enabled: bool, model_name: Optional[str]
) -> Optional[Reranker]:
    """Return a reranker or None (None => skip the rerank step entirely)."""
    if not enabled:
        return None
    if model_name:
        try:
            return CrossEncoderReranker(model_name)
        except ImportError:
            # Fall back rather than crash: ordering still happens, just lexically.
            return LexicalReranker()
    return LexicalReranker()
