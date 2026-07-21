"""Pro-tier document store: dedicated Postgres + pgvector.

Holds document chunks and their embeddings with an HNSW index. Facts live in
the Graphiti-backed graph store; `ProStore` (in factory.py) composes the two so
the engine sees a single `MemoryStore`.

Requires a running Postgres with the pgvector extension; not exercised by the
offline test suite. Behind lazy imports so the core installs without psycopg.
"""

from __future__ import annotations

from typing import Sequence

from ..errors import StoreError
from ..types import Chunk
from .base import GraphHit, VectorHit


class PgVectorStore:
    """Document chunks in Postgres/pgvector. Facts are handled elsewhere."""

    def __init__(self, url: str, embed_dim: int = 256, table_prefix: str = "dmem"):
        try:
            import psycopg  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise StoreError(
                "Pro tier needs the 'pgvector' extra: pip install dmem[pgvector]"
            ) from e
        self._psycopg = __import__("psycopg")
        self._url = url
        self._dim = embed_dim
        # Validate the prefix (used in identifiers, so keep it strict).
        import re
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_prefix):
            raise StoreError(f"Invalid PGVECTOR_TABLE_PREFIX: {table_prefix!r}")
        self._chunks = f"{table_prefix}_chunks"
        self._meta = f"{table_prefix}_meta"
        self._conn = None

    def _connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = self._psycopg.connect(self._url, autocommit=True)
        return self._conn

    def initialize(self) -> None:
        conn = self._connect()
        t = self._chunks
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {t} (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    concept TEXT, section TEXT, ordinal INT NOT NULL,
                    text TEXT NOT NULL, content_hash TEXT NOT NULL,
                    embedding vector({self._dim}),
                    metadata JSONB, provenance JSONB)""")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_ns ON {t}(namespace)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_hash "
                         f"ON {t}(namespace, content_hash)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_hnsw ON {t} "
                         f"USING hnsw (embedding vector_cosine_ops)")
            conn.execute(f"CREATE TABLE IF NOT EXISTS {self._meta} "
                         f"(key TEXT PRIMARY KEY, value TEXT)")
        except Exception as e:  # pragma: no cover - server dependent
            raise StoreError(f"pgvector initialize failed: {e}") from e

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        import json
        conn = self._connect()
        for c in chunks:
            conn.execute(
                f"""INSERT INTO {self._chunks} (id, namespace, document_id, concept,
                    section, ordinal, text, content_hash, embedding, metadata,
                    provenance)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text,
                    embedding=EXCLUDED.embedding""",
                (c.id, c.namespace, c.document_id, c.concept, c.section,
                 c.ordinal, c.text, _hash(c.text), _vec(c.embedding),
                 json.dumps(c.metadata),
                 json.dumps(c.provenance.to_dict()) if c.provenance else None))

    def chunk_exists(self, namespace: str, content_hash: str) -> bool:
        conn = self._connect()
        row = conn.execute(
            f"SELECT 1 FROM {self._chunks} WHERE namespace=%s AND content_hash=%s "
            f"LIMIT 1", (namespace, content_hash)).fetchone()
        return row is not None

    def vector_search(self, namespace, query_embedding, top_k,
                      kinds=("chunk",)) -> list[VectorHit]:
        if "chunk" not in kinds:
            return []
        conn = self._connect()
        rows = conn.execute(
            f"""SELECT id, text, document_id, concept, section,
                      1 - (embedding <=> %s) AS score
               FROM {self._chunks} WHERE namespace=%s
               ORDER BY embedding <=> %s LIMIT %s""",
            (_vec(query_embedding), namespace, _vec(query_embedding), top_k)
        ).fetchall()
        return [VectorHit(id=r[0], text=r[1], score=float(r[5]), kind="chunk",
                          payload={"document_id": r[2], "concept": r[3],
                                   "section": r[4]}) for r in rows]

    def keyword_search(self, namespace, query, top_k,
                       kinds=("chunk",)) -> list[VectorHit]:
        if "chunk" not in kinds:
            return []
        conn = self._connect()
        rows = conn.execute(
            f"""SELECT id, text, document_id, concept, section,
                      ts_rank(to_tsvector('english', text),
                              plainto_tsquery('english', %s)) AS rank
               FROM {self._chunks}
               WHERE namespace=%s AND to_tsvector('english', text)
                     @@ plainto_tsquery('english', %s)
               ORDER BY rank DESC LIMIT %s""",
            (query, namespace, query, top_k)).fetchall()
        return [VectorHit(id=r[0], text=r[1], score=float(r[5]), kind="chunk",
                          payload={"document_id": r[2], "concept": r[3],
                                   "section": r[4]}) for r in rows]

    def graph_neighbors(self, namespace, seeds, max_hops=1, limit=20) -> list[GraphHit]:
        return []  # documents have no graph edges; facts handle this

    # -- meta (table created in initialize) --------------------------------
    def get_meta(self, key: str):
        conn = self._connect()
        row = conn.execute(f"SELECT value FROM {self._meta} WHERE key=%s",
                           (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._connect().execute(
            f"INSERT INTO {self._meta} (key, value) VALUES (%s,%s) "
            f"ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", (key, value))

    def iter_chunks(self):
        conn = self._connect()
        rows = conn.execute(
            f"SELECT id, namespace, document_id, concept, section, ordinal, text "
            f"FROM {self._chunks}").fetchall()
        for r in rows:
            yield Chunk(text=r[6], document_id=r[2], concept=r[3], section=r[4],
                        ordinal=r[5], id=r[0], namespace=r[1])

    def delete_namespace(self, namespace: str) -> int:
        conn = self._connect()
        cur = conn.execute(f"DELETE FROM {self._chunks} WHERE namespace=%s",
                           (namespace,))
        return cur.rowcount or 0


def _vec(embedding) -> str:
    if embedding is None:
        return None  # type: ignore[return-value]
    return "[" + ",".join(str(x) for x in embedding) + "]"


def _hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
