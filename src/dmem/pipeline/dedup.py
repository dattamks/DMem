"""Ingestion dedup helpers.

Two levels:
- Document chunks: a content hash keyed by namespace, checked before insert so
  re-uploading or re-syncing the same document doesn't bloat the store.
- Facts: dedup is by the fact's (namespace, subject, predicate, object) key —
  see `Fact.dedup_key()`. The engine uses this together with contradiction
  handling (same subject+predicate, *different* object => supersede, not dup).
"""

from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
