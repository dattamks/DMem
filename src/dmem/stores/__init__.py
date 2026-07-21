"""Backing stores across the three tiers.

All tiers implement the same `MemoryStore` protocol so the engine, and every
distribution surface, are backend-agnostic. Retrieval quality differs by tier;
the *interface* does not.

- SQLite tier    -> `SQLiteStore`            (vectors + validity-window facts)
- Consolidated   -> `GraphStore`             (one graph DB, native vectors)
- Pro tier       -> `ProStore`               (pgvector docs + graph facts)
"""

from .base import MemoryStore, VectorHit, GraphHit
from .factory import build_store

__all__ = ["MemoryStore", "VectorHit", "GraphHit", "build_store"]
