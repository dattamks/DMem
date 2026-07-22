"""Hybrid retriever: runs the three channels, fuses, recency-weights, reranks.

Pipeline (see scoring.py for the ordering rationale):

    vector  ┐
    keyword ├─> RRF fusion ─> recency weight ─> conflict flag ─> rerank ─> cut
    graph   ┘

The three channels are independent and could be parallelized; the SQLite tier
runs them inline (cheap). Higher tiers may push this concurrency down later.
"""

from __future__ import annotations

import time
from typing import Optional

from ..providers.embeddings import EmbeddingProvider
from ..providers.reranker import Reranker
from ..stores.base import MemoryStore
from ..types import RetrievalResult
from .scoring import apply_recency, detect_conflicts, reciprocal_rank_fusion


class HybridRetriever:
    def __init__(self, store: MemoryStore, embedder: EmbeddingProvider,
                 reranker: Optional[Reranker], *, top_k: int = 20,
                 rerank_top_k: int = 8, recency_half_life_days: float = 30.0):
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k
        self._half_life = recency_half_life_days

    def retrieve(self, namespace: str, query: str, *,
                 kinds: tuple[str, ...] = ("fact", "chunk"),
                 seeds: Optional[list[str]] = None,
                 now: Optional[float] = None) -> list[RetrievalResult]:
        now = time.time() if now is None else now
        q_emb = self._embedder.embed_one(query)

        vector_hits = self._store.vector_search(namespace, q_emb, self._top_k, kinds)
        keyword_hits = self._store.keyword_search(namespace, query, self._top_k, kinds)
        graph_hits = []
        if "fact" in kinds:
            seed_terms = seeds or _query_terms(query)
            graph_hits = self._store.graph_neighbors(namespace, seed_terms,
                                                     max_hops=1, limit=self._top_k)

        fused = reciprocal_rank_fusion(vector_hits, keyword_hits, graph_hits)
        if not fused:
            return []

        apply_recency(fused, self._half_life, now)
        detect_conflicts(fused)

        ordered = sorted(fused.values(), key=lambda r: r.score, reverse=True)
        candidates = ordered[: max(self._rerank_top_k * 3, self._rerank_top_k)]

        if self._reranker is not None and candidates:
            scores = self._reranker.rerank(query, [c.text for c in candidates])
            for c, s in zip(candidates, scores):
                # Preserve recency signal: blend rerank relevance with prior score.
                c.payload["rerank_score"] = float(s)
                c.score = 0.7 * float(s) + 0.3 * c.score
            candidates.sort(key=lambda r: r.score, reverse=True)

        return candidates[: self._rerank_top_k]


def _query_terms(query: str) -> list[str]:
    import re
    # crude entity seeds: capitalized words + all tokens, deduped
    caps = re.findall(r"[A-Z][\w'-]+", query)
    toks = re.findall(r"[a-zA-Z0-9]{3,}", query)
    seen, out = set(), []
    for t in caps + toks:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            out.append(t)
    return out[:10]
