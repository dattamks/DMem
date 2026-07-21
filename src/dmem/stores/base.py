"""The `MemoryStore` protocol every tier implements.

Splitting responsibilities into three retrieval channels — vector, keyword,
graph — is deliberate: the engine fuses them (hybrid retrieval). A tier that
cannot do true graph traversal (SQLite) still implements ``graph_neighbors`` in
a degraded form so the fusion code path is identical everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

from ..types import Chunk, Fact


@dataclass
class VectorHit:
    id: str
    text: str
    score: float          # cosine similarity in [-1, 1] (or provider-native)
    kind: str             # "fact" | "chunk"
    payload: dict


@dataclass
class GraphHit:
    id: str
    text: str
    hops: int
    payload: dict


@runtime_checkable
class MemoryStore(Protocol):
    tier: str

    # -- lifecycle ---------------------------------------------------------
    def initialize(self) -> None: ...
    def close(self) -> None: ...

    # -- facts -------------------------------------------------------------
    def upsert_fact(self, fact: Fact) -> None:
        """Insert a fact. Callers handle contradiction/supersede logic first."""

    def find_fact_by_dedup_key(self, namespace: str, dedup_key: str) -> Optional[Fact]:
        ...

    def find_current_facts(
        self, namespace: str, subject: str, predicate: str
    ) -> list[Fact]:
        """Current (valid_to is None) facts matching subject+predicate.

        Used to detect contradictions before inserting a new value."""

    def close_fact(self, fact_id: str, valid_to: float) -> None:
        """Close a fact's validity window (bi-temporal). Never deletes."""

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        ...

    def delete_fact(self, fact_id: str) -> bool:
        """Hard-delete a single fact by id. Returns True if a row was removed."""

    def delete_facts(
        self, namespace: str, *, subject: Optional[str] = None,
        predicate: Optional[str] = None, object: Optional[str] = None,
    ) -> int:
        """Hard-delete all facts (current and closed) matching the filters.

        Right-to-be-forgotten at fact granularity, as distinct from the
        preserve-history close. At least one filter must be non-None."""

    # -- documents ---------------------------------------------------------
    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None: ...
    def chunk_exists(self, namespace: str, content_hash: str) -> bool: ...

    # -- retrieval channels ------------------------------------------------
    def vector_search(
        self, namespace: str, query_embedding: list[float], top_k: int,
        kinds: Sequence[str] = ("fact", "chunk"),
    ) -> list[VectorHit]:
        ...

    def keyword_search(
        self, namespace: str, query: str, top_k: int,
        kinds: Sequence[str] = ("fact", "chunk"),
    ) -> list[VectorHit]:
        ...

    def graph_neighbors(
        self, namespace: str, seeds: Sequence[str], max_hops: int = 1,
        limit: int = 20,
    ) -> list[GraphHit]:
        """Expand from seed entities across relationships.

        Pro/consolidated tiers do real multi-hop traversal. SQLite approximates
        with a one-hop subject/object co-occurrence lookup."""

    # -- admin -------------------------------------------------------------
    def delete_namespace(self, namespace: str) -> int:
        """Hard-delete everything for a namespace (GDPR / right-to-be-forgotten).

        Bi-temporal history is preserved for *contradictions*, but a namespace
        wipe is an explicit, irreversible operation that removes it all."""
