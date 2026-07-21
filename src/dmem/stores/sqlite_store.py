"""SQLite tier — the zero-setup default.

- Vectors: stored as JSON, brute-force cosine similarity in Python (numpy used
  when the ``fast`` extra is present). This is O(n) per query, which is fine for
  the personal/single-project scale this tier targets. The ``consolidated`` and
  ``pro`` tiers exist precisely for larger corpora with ANN indexes.
- Facts: a plain table with bi-temporal ``valid_from`` / ``valid_to`` columns.
  Contradictions are handled by the engine, which closes the old row's window
  and inserts a new row — nothing is deleted.
- Graph: approximated. ``graph_neighbors`` does a one-hop co-occurrence lookup
  over shared subjects/objects, so the fusion code path is identical to the
  higher tiers even though there is no real graph engine here.

No server. No third-party runtime dependency.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from typing import Optional, Sequence

from ..errors import StoreError
from ..types import Chunk, Fact, FactType, Provenance
from .base import GraphHit, VectorHit

_TOKEN_RE = re.compile(r"[a-z0-9]+")

try:  # optional acceleration
    import numpy as _np  # type: ignore
except ImportError:  # pragma: no cover
    _np = None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    if _np is not None:
        va, vb = _np.asarray(a), _np.asarray(b)
        na, nb = _np.linalg.norm(va), _np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(va.dot(vb) / (na * nb))
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SQLiteStore:
    tier = "sqlite"

    def __init__(self, path: str = "dmem.db"):
        self.path = path
        # check_same_thread=False + a lock: the MCP server and IDE extension may
        # call from different threads.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()

    # -- lifecycle ---------------------------------------------------------
    def initialize(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    dedup_key TEXT NOT NULL,
                    embedding TEXT,
                    valid_from REAL NOT NULL,
                    valid_to REAL,
                    recorded_at REAL NOT NULL,
                    supersedes TEXT,
                    provenance TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_facts_ns ON facts(namespace);
                CREATE INDEX IF NOT EXISTS idx_facts_dedup ON facts(namespace, dedup_key);
                CREATE INDEX IF NOT EXISTS idx_facts_sp
                    ON facts(namespace, subject, predicate, valid_to);

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    concept TEXT,
                    section TEXT,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding TEXT,
                    metadata TEXT,
                    provenance TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_ns ON chunks(namespace);
                CREATE INDEX IF NOT EXISTS idx_chunks_hash
                    ON chunks(namespace, content_hash);
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- facts -------------------------------------------------------------
    def upsert_fact(self, fact: Fact) -> None:
        prov = json.dumps(fact.provenance.to_dict()) if fact.provenance else None
        emb = json.dumps(fact.embedding) if fact.embedding else None
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO facts
                   (id, namespace, subject, predicate, object, fact_type,
                    confidence, dedup_key, embedding, valid_from, valid_to,
                    recorded_at, supersedes, provenance)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    fact.id, fact.namespace, fact.subject, fact.predicate,
                    fact.object, fact.fact_type.value, fact.confidence,
                    fact.dedup_key(), emb, fact.valid_from, fact.valid_to,
                    fact.recorded_at, fact.supersedes, prov,
                ),
            )
            self._conn.commit()

    def _row_to_fact(self, r: sqlite3.Row) -> Fact:
        prov = json.loads(r["provenance"]) if r["provenance"] else None
        return Fact(
            subject=r["subject"], predicate=r["predicate"], object=r["object"],
            fact_type=FactType.coerce(r["fact_type"]),
            provenance=Provenance.from_dict(prov) if prov else None,
            id=r["id"], namespace=r["namespace"], confidence=r["confidence"],
            embedding=json.loads(r["embedding"]) if r["embedding"] else None,
            valid_from=r["valid_from"], valid_to=r["valid_to"],
            recorded_at=r["recorded_at"], supersedes=r["supersedes"],
        )

    def find_fact_by_dedup_key(self, namespace: str, dedup_key: str) -> Optional[Fact]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM facts WHERE namespace=? AND dedup_key=? "
                "AND valid_to IS NULL LIMIT 1",
                (namespace, dedup_key),
            ).fetchone()
        return self._row_to_fact(row) if row else None

    def find_current_facts(
        self, namespace: str, subject: str, predicate: str
    ) -> list[Fact]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE namespace=? AND lower(subject)=? "
                "AND lower(predicate)=? AND valid_to IS NULL",
                (namespace, subject.strip().lower(), predicate.strip().lower()),
            ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def close_fact(self, fact_id: str, valid_to: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE facts SET valid_to=? WHERE id=? AND valid_to IS NULL",
                (valid_to, fact_id),
            )
            self._conn.commit()

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM facts WHERE id=?", (fact_id,)
            ).fetchone()
        return self._row_to_fact(row) if row else None

    # -- documents ---------------------------------------------------------
    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            for c in chunks:
                prov = json.dumps(c.provenance.to_dict()) if c.provenance else None
                emb = json.dumps(c.embedding) if c.embedding else None
                self._conn.execute(
                    """INSERT OR REPLACE INTO chunks
                       (id, namespace, document_id, concept, section, ordinal,
                        text, content_hash, embedding, metadata, provenance)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        c.id, c.namespace, c.document_id, c.concept, c.section,
                        c.ordinal, c.text, _content_hash(c.text), emb,
                        json.dumps(c.metadata), prov,
                    ),
                )
            self._conn.commit()

    def chunk_exists(self, namespace: str, content_hash: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM chunks WHERE namespace=? AND content_hash=? LIMIT 1",
                (namespace, content_hash),
            ).fetchone()
        return row is not None

    # -- retrieval channels ------------------------------------------------
    def _iter_embedded(self, namespace: str, kinds: Sequence[str]):
        with self._lock:
            if "fact" in kinds:
                for r in self._conn.execute(
                    "SELECT id, subject, predicate, object, embedding, fact_type, "
                    "confidence, valid_to, recorded_at, valid_from FROM facts "
                    "WHERE namespace=? AND embedding IS NOT NULL",
                    (namespace,),
                ).fetchall():
                    yield ("fact", r)
            if "chunk" in kinds:
                for r in self._conn.execute(
                    "SELECT id, text, embedding, document_id, concept, section "
                    "FROM chunks WHERE namespace=? AND embedding IS NOT NULL",
                    (namespace,),
                ).fetchall():
                    yield ("chunk", r)

    def vector_search(
        self, namespace: str, query_embedding: list[float], top_k: int,
        kinds: Sequence[str] = ("fact", "chunk"),
    ) -> list[VectorHit]:
        hits: list[VectorHit] = []
        for kind, r in self._iter_embedded(namespace, kinds):
            emb = json.loads(r["embedding"])
            score = _cosine(query_embedding, emb)
            if kind == "fact":
                text = f"{r['subject']} {r['predicate']} {r['object']}".strip()
                payload = {
                    "fact_type": r["fact_type"], "confidence": r["confidence"],
                    "valid_to": r["valid_to"], "recorded_at": r["recorded_at"],
                    "valid_from": r["valid_from"], "subject": r["subject"],
                    "predicate": r["predicate"], "object": r["object"],
                }
            else:
                text = r["text"]
                payload = {
                    "document_id": r["document_id"], "concept": r["concept"],
                    "section": r["section"],
                }
            hits.append(VectorHit(id=r["id"], text=text, score=score,
                                  kind=kind, payload=payload))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def keyword_search(
        self, namespace: str, query: str, top_k: int,
        kinds: Sequence[str] = ("fact", "chunk"),
    ) -> list[VectorHit]:
        tokens = set(_TOKEN_RE.findall(query.lower()))
        if not tokens:
            return []
        hits: list[VectorHit] = []
        with self._lock:
            if "fact" in kinds:
                rows = self._conn.execute(
                    "SELECT id, subject, predicate, object, fact_type, confidence, "
                    "valid_to, recorded_at, valid_from FROM facts WHERE namespace=?",
                    (namespace,),
                ).fetchall()
                for r in rows:
                    text = f"{r['subject']} {r['predicate']} {r['object']}".strip()
                    score = _token_overlap(tokens, text)
                    if score > 0:
                        hits.append(VectorHit(
                            id=r["id"], text=text, score=score, kind="fact",
                            payload={
                                "fact_type": r["fact_type"],
                                "confidence": r["confidence"],
                                "valid_to": r["valid_to"],
                                "recorded_at": r["recorded_at"],
                                "valid_from": r["valid_from"],
                                "subject": r["subject"], "predicate": r["predicate"],
                                "object": r["object"],
                            }))
            if "chunk" in kinds:
                rows = self._conn.execute(
                    "SELECT id, text, document_id, concept, section FROM chunks "
                    "WHERE namespace=?", (namespace,),
                ).fetchall()
                for r in rows:
                    score = _token_overlap(tokens, r["text"])
                    if score > 0:
                        hits.append(VectorHit(
                            id=r["id"], text=r["text"], score=score, kind="chunk",
                            payload={"document_id": r["document_id"],
                                     "concept": r["concept"], "section": r["section"]}))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def graph_neighbors(
        self, namespace: str, seeds: Sequence[str], max_hops: int = 1,
        limit: int = 20,
    ) -> list[GraphHit]:
        """One-hop co-occurrence approximation of graph traversal.

        Given seed entity strings, find current facts whose subject or object
        matches a seed, returning the *other* end as a neighbor. Real multi-hop
        traversal is a higher-tier capability."""
        if not seeds:
            return []
        seed_set = {s.strip().lower() for s in seeds if s}
        hits: list[GraphHit] = []
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, subject, predicate, object FROM facts "
                "WHERE namespace=? AND valid_to IS NULL", (namespace,),
            ).fetchall()
        for r in rows:
            subj, obj = r["subject"].strip().lower(), r["object"].strip().lower()
            if subj in seed_set or obj in seed_set:
                text = f"{r['subject']} {r['predicate']} {r['object']}".strip()
                hits.append(GraphHit(id=r["id"], text=text, hops=1,
                                     payload={"subject": r["subject"],
                                              "predicate": r["predicate"],
                                              "object": r["object"]}))
                if len(hits) >= limit:
                    break
        return hits

    # -- admin -------------------------------------------------------------
    def delete_namespace(self, namespace: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM facts WHERE namespace=?", (namespace,))
            n = cur.rowcount
            cur2 = self._conn.execute(
                "DELETE FROM chunks WHERE namespace=?", (namespace,))
            self._conn.commit()
            return n + cur2.rowcount


def _token_overlap(query_tokens: set[str], text: str) -> float:
    ttoks = _TOKEN_RE.findall(text.lower())
    if not ttoks:
        return 0.0
    tset = set(ttoks)
    overlap = len(query_tokens & tset)
    if overlap == 0:
        return 0.0
    return overlap / (math.sqrt(len(tset)) + 1.0)


def _content_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
