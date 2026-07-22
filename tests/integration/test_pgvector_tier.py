"""Real pgvector integration — no Docker required.

Uses `pgserver` (pip-installable; bundles PostgreSQL + pgvector binaries) to spin
up a genuine local Postgres and exercise the pro-tier document store end to end.
Skips when pgserver/psycopg aren't installed:

    pip install -e ".[pgvector,dev]" pgserver
    pytest tests/integration/test_pgvector_tier.py -q
"""

import hashlib

import pytest

pgserver = pytest.importorskip("pgserver")
pytest.importorskip("psycopg")

from dmem.stores.pgvector_store import PgVectorStore, ensure_pgvector
from dmem.types import Chunk


@pytest.fixture(scope="module")
def pg_uri(tmp_path_factory):
    pgdata = tmp_path_factory.mktemp("dmem-pg")
    server = pgserver.get_server(str(pgdata))
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


def test_ensure_pgvector_enables_extension(pg_uri):
    import psycopg
    with psycopg.connect(pg_uri, autocommit=True) as conn:
        state = ensure_pgvector(conn)
        assert state in ("created", "enabled")
        row = conn.execute(
            "SELECT 1 FROM pg_extension WHERE extname='vector'").fetchone()
        assert row is not None  # actually enabled now


def test_pgvector_store_round_trip(pg_uri):
    store = PgVectorStore(pg_uri, embed_dim=4, table_prefix="rt")
    store.initialize()
    try:
        store.upsert_chunks([
            Chunk(text="Prod runs Postgres 16 in AWS us-east-1",
                  document_id="runbook", namespace="t1",
                  embedding=[0.1, 0.2, 0.3, 0.4]),
            Chunk(text="Nightly backups retained 30 days in S3",
                  document_id="runbook", namespace="t1",
                  embedding=[0.9, 0.1, 0.0, 0.0]),
        ])
        hits = store.vector_search("t1", [0.1, 0.2, 0.3, 0.4], top_k=2)
        assert hits and "us-east-1" in hits[0].text
        assert hits[0].score > 0.99  # exact-match cosine

        kw = store.keyword_search("t1", "backups", top_k=2)
        assert kw and "backups" in kw[0].text.lower()

        h = hashlib.sha256(
            b"Prod runs Postgres 16 in AWS us-east-1").hexdigest()
        assert store.chunk_exists("t1", h) is True
    finally:
        store.close()


def test_pgvector_namespace_isolation_and_delete(pg_uri):
    store = PgVectorStore(pg_uri, embed_dim=3, table_prefix="iso")
    store.initialize()
    try:
        store.upsert_chunks([Chunk(text="alpha", document_id="d", namespace="a",
                                   embedding=[1.0, 0.0, 0.0])])
        store.upsert_chunks([Chunk(text="beta", document_id="d", namespace="b",
                                   embedding=[0.0, 1.0, 0.0])])
        a = store.vector_search("a", [1.0, 0.0, 0.0], top_k=5)
        assert [h.text for h in a] == ["alpha"]  # no cross-namespace bleed

        removed = store.delete_namespace("a")
        assert removed == 1
        assert store.vector_search("a", [1.0, 0.0, 0.0], top_k=5) == []
        assert store.vector_search("b", [0.0, 1.0, 0.0], top_k=5)  # b intact
    finally:
        store.close()


def test_pgvector_meta_roundtrip(pg_uri):
    store = PgVectorStore(pg_uri, embed_dim=2, table_prefix="meta")
    store.initialize()
    try:
        assert store.get_meta("embedding_signature") is None
        store.set_meta("embedding_signature", "openai:model:1536")
        assert store.get_meta("embedding_signature") == "openai:model:1536"
        store.set_meta("embedding_signature", "changed")  # upsert
        assert store.get_meta("embedding_signature") == "changed"
    finally:
        store.close()


def test_table_prefix_creates_isolated_tables(pg_uri):
    import psycopg
    PgVectorStore(pg_uri, embed_dim=2, table_prefix="tenantx").initialize()
    with psycopg.connect(pg_uri, autocommit=True) as conn:
        row = conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name='tenantx_chunks'").fetchone()
        assert row is not None
