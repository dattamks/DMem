"""Build the right `MemoryStore` for the configured tier.

- sqlite       -> SQLiteStore
- consolidated -> GraphStore(hold_chunks=True)
- pro          -> ProStore(GraphStore facts + PgVectorStore documents)

The `ProStore` composite makes the two pro-tier servers look like one store to
the engine, routing facts to the graph and chunks to pgvector.
"""

from __future__ import annotations

from typing import Sequence

from ..config import Config, GraphKind, Tier
from .base import GraphHit, MemoryStore, VectorHit


def build_store(cfg: Config, embed_dim: int) -> MemoryStore:
    if cfg.tier is Tier.SQLITE:
        from .sqlite_store import SQLiteStore
        return SQLiteStore(cfg.sqlite_path)

    if cfg.tier is Tier.CONSOLIDATED:
        # Kuzu is embedded (a path, not a network server); it holds both facts
        # and chunks in-process. Neo4j/FalkorDB are networked servers.
        if cfg.graph.kind is GraphKind.KUZU:
            from .kuzu_store import KuzuGraphStore
            path = cfg.graph.url or "dmem_graph.kz"
            return KuzuGraphStore(path, embed_dim=embed_dim)
        from .graph_store import GraphStore
        return GraphStore(cfg.graph, hold_chunks=True, embed_dim=embed_dim)

    if cfg.tier is Tier.PRO:
        from .graph_store import GraphStore
        from .pgvector_store import PgVectorStore
        graph = GraphStore(cfg.graph, hold_chunks=False, embed_dim=embed_dim)
        docs = PgVectorStore(cfg.pgvector_url, embed_dim=embed_dim,  # type: ignore[arg-type]
                             table_prefix=cfg.pgvector_table_prefix)
        return ProStore(graph, docs)

    raise ValueError(f"Unknown tier: {cfg.tier}")


class ProStore:
    """Composite: facts/relationships in the graph, documents in pgvector."""

    tier = "pro"

    def __init__(self, graph, docs):
        self._graph = graph
        self._docs = docs

    def initialize(self) -> None:
        self._graph.initialize()
        self._docs.initialize()

    def close(self) -> None:
        self._graph.close()
        self._docs.close()

    # facts -> graph
    def upsert_fact(self, fact): self._graph.upsert_fact(fact)
    def find_fact_by_dedup_key(self, ns, dk): return self._graph.find_fact_by_dedup_key(ns, dk)
    def find_current_facts(self, ns, s, p): return self._graph.find_current_facts(ns, s, p)
    def close_fact(self, fid, vto): self._graph.close_fact(fid, vto)
    def get_fact(self, fid): return self._graph.get_fact(fid)
    def delete_fact(self, fid): return self._graph.delete_fact(fid)
    def delete_facts(self, ns, *, subject=None, predicate=None, object=None):
        return self._graph.delete_facts(ns, subject=subject, predicate=predicate,
                                        object=object)

    # documents -> pgvector
    def upsert_chunks(self, chunks): self._docs.upsert_chunks(chunks)
    def chunk_exists(self, ns, h): return self._docs.chunk_exists(ns, h)

    # retrieval: facts from graph, chunks from pgvector, merged
    def vector_search(self, ns, q, top_k, kinds=("fact", "chunk")) -> list[VectorHit]:
        hits: list[VectorHit] = []
        if "fact" in kinds:
            hits += self._graph.vector_search(ns, q, top_k, kinds=("fact",))
        if "chunk" in kinds:
            hits += self._docs.vector_search(ns, q, top_k, kinds=("chunk",))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def keyword_search(self, ns, query, top_k, kinds=("fact", "chunk")) -> list[VectorHit]:
        hits: list[VectorHit] = []
        if "fact" in kinds:
            hits += self._graph.keyword_search(ns, query, top_k, kinds=("fact",))
        if "chunk" in kinds:
            hits += self._docs.keyword_search(ns, query, top_k, kinds=("chunk",))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def graph_neighbors(self, ns, seeds, max_hops=1, limit=20) -> list[GraphHit]:
        return self._graph.graph_neighbors(ns, seeds, max_hops, limit)

    # meta canonical in the graph store; chunks iterate from pgvector
    def get_meta(self, key): return self._graph.get_meta(key)
    def set_meta(self, key, value): self._graph.set_meta(key, value)
    def iter_facts(self): return self._graph.iter_facts()
    def iter_chunks(self): return self._docs.iter_chunks()

    def delete_namespace(self, ns) -> int:
        return self._graph.delete_namespace(ns) + self._docs.delete_namespace(ns)
