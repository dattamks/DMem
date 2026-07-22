"""Scoring utilities: fusion, recency weighting, and conflict detection.

Ordering decision (called out as an open question in the source spec): DMem
applies recency weighting to the *fused* score BEFORE reranking selects the
final set, but reranking only re-orders the top candidates by semantic
relevance — it does not undo recency. Concretely the pipeline is:

    fuse(vector, keyword, graph)  ->  recency-weight  ->  rerank top-N  ->  cut

This keeps a stale fact from outranking a newer contradicting one (recency
matters at candidate-selection time) while still letting the cross-encoder pick
the most *relevant* items from the recency-aware candidate pool. The weights are
day-one defaults; calibration against real traffic is deferred per the spec.
"""

from __future__ import annotations

import math
from typing import Iterable

from ..stores.base import GraphHit, VectorHit
from ..types import RetrievalResult

# Reciprocal-rank-fusion constant.
_RRF_K = 60

# Channel weights for fusion (day-one defaults).
_CHANNEL_WEIGHTS = {"vector": 1.0, "keyword": 0.6, "graph": 0.8}


def reciprocal_rank_fusion(
    vector_hits: list[VectorHit],
    keyword_hits: list[VectorHit],
    graph_hits: list[GraphHit],
) -> dict[str, RetrievalResult]:
    """Fuse the three channels via weighted Reciprocal Rank Fusion.

    RRF is robust to the different score scales of cosine similarity, lexical
    overlap, and graph proximity — we fuse by *rank*, not by raw score."""
    fused: dict[str, RetrievalResult] = {}

    def add(hits: Iterable, channel: str, kind_default: str = "fact"):
        for rank, h in enumerate(hits):
            contrib = _CHANNEL_WEIGHTS[channel] / (_RRF_K + rank + 1)
            hid = h.id
            if hid not in fused:
                kind = getattr(h, "kind", kind_default)
                payload = dict(getattr(h, "payload", {}) or {})
                payload["channels"] = []
                fused[hid] = RetrievalResult(
                    kind=kind, text=h.text, score=0.0, source_id=hid,
                    payload=payload)
            fused[hid].score += contrib
            fused[hid].payload["channels"].append(channel)

    add(vector_hits, "vector")
    add(keyword_hits, "keyword")
    add(graph_hits, "graph")
    return fused


def apply_recency(results: dict[str, RetrievalResult], half_life_days: float,
                  now: float) -> None:
    """Multiply each result's score by a recency decay factor in-place.

    Uses the fact's ``recorded_at`` (falls back to ``valid_from``). Chunks
    without a timestamp are left unweighted (neutral factor 1.0)."""
    half_life_s = half_life_days * 86400.0
    for r in results.values():
        ts = r.payload.get("recorded_at") or r.payload.get("valid_from")
        if ts is None:
            continue
        age = max(0.0, now - float(ts))
        decay = math.pow(0.5, age / half_life_s) if half_life_s > 0 else 1.0
        # Blend so recency nudges rather than dominates: 0.5 floor.
        r.score *= (0.5 + 0.5 * decay)


def detect_conflicts(results: dict[str, RetrievalResult]) -> list[dict]:
    """Flag facts that share subject+predicate but differ in object.

    Explicit conflict surfacing: when a fact has changed we mark it, rather than
    silently serving only the newest value. Returns a list of conflict records
    and annotates the involved results' ``conflict`` field."""
    by_sp: dict[tuple[str, str], list[RetrievalResult]] = {}
    for r in results.values():
        if r.kind != "fact":
            continue
        subj = str(r.payload.get("subject", "")).strip().lower()
        pred = str(r.payload.get("predicate", "")).strip().lower()
        if not subj or not pred:
            continue
        by_sp.setdefault((subj, pred), []).append(r)

    conflicts: list[dict] = []
    for (subj, pred), group in by_sp.items():
        objects = {str(r.payload.get("object", "")).strip().lower() for r in group}
        if len(objects) <= 1:
            continue
        # A genuine "changed fact" is one where a prior value was actually
        # superseded — i.e. at least one member has a closed validity window.
        # Multiple concurrent values of a multi-valued predicate (all current)
        # are NOT a conflict, just co-existing facts.
        has_closed = any(r.payload.get("valid_to") not in (None, "") for r in group)
        if not has_closed:
            continue
        # Newest current value vs older/closed ones.
        group.sort(key=lambda r: float(r.payload.get("recorded_at") or 0),
                   reverse=True)
        current = [r for r in group if r.payload.get("valid_to") in (None, "")]
        record = {
            "subject": subj, "predicate": pred,
            "values": [{
                "object": r.payload.get("object"),
                "recorded_at": r.payload.get("recorded_at"),
                "is_current": r.payload.get("valid_to") in (None, ""),
            } for r in group],
            "note": "This fact changed over time — multiple values on record.",
        }
        conflicts.append(record)
        for r in group:
            r.conflict = record
    return conflicts
