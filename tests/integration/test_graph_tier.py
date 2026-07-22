"""End-to-end integration tests for the graph-backed consolidated tier.

These run ONLY when a live graph server is configured, so the offline suite is
unaffected. Point them at a local FalkorDB or Neo4j:

    # FalkorDB (lightweight Redis module):
    docker run -d -p 6379:6379 falkordb/falkordb
    export DMEM_TEST_GRAPH_KIND=falkordb
    export DMEM_TEST_GRAPH_URL=redis://localhost:6379
    pip install "dmem[falkordb,dev]" && pytest tests/integration -q

    # Neo4j 5.11+:
    export DMEM_TEST_GRAPH_KIND=neo4j
    export DMEM_TEST_GRAPH_URL=bolt://localhost:7687
    export DMEM_TEST_GRAPH_USER=neo4j DMEM_TEST_GRAPH_PASSWORD=password

They exercise the full engine against the real backend: ingest -> typed facts,
single-valued contradiction (supersede, not delete), multi-valued accumulation,
hybrid retrieval, handoff, and GDPR deletes. A unique namespace per run keeps
them isolated; everything is torn down at the end.
"""

import os
import uuid
import warnings

import pytest

_URL = os.environ.get("DMEM_TEST_GRAPH_URL")
_KIND = os.environ.get("DMEM_TEST_GRAPH_KIND")

pytestmark = pytest.mark.skipif(
    not (_URL and _KIND),
    reason="set DMEM_TEST_GRAPH_URL and DMEM_TEST_GRAPH_KIND to run graph "
           "integration tests")


@pytest.fixture
def engine():
    from dmem import Config, DMemEngine, Tier
    from dmem.config import EmbeddingConfig, GraphConfig, GraphKind
    cfg = Config(
        tier=Tier.CONSOLIDATED,
        embedding=EmbeddingConfig(dim=256),  # offline hashing embedder
        graph=GraphConfig(
            kind=GraphKind(_KIND),
            url=_URL,
            user=os.environ.get("DMEM_TEST_GRAPH_USER"),
            password=os.environ.get("DMEM_TEST_GRAPH_PASSWORD"),
            database=os.environ.get("DMEM_TEST_GRAPH_DATABASE"),
        ),
    )
    ns = f"itest_{uuid.uuid4().hex[:8]}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = DMemEngine(cfg)
    yield eng, ns
    try:
        eng.forget(ns)
    finally:
        eng.close()


def test_ingest_and_retrieve(engine):
    eng, ns = engine
    eng.ingest_message("My name is Ada and I work at Analytical Engines.",
                       namespace=ns)
    results = eng.retrieve("where does the user work?", namespace=ns)
    assert any("analytical engines" in r.text.lower() for r in results)


def test_single_valued_contradiction(engine):
    eng, ns = engine
    eng.ingest_message("I work at Google.", namespace=ns)
    eng.ingest_message("I work at Meta.", namespace=ns)
    current = eng.store.find_current_facts(ns, "user", "works_at")
    objs = {f.object.lower() for f in current}
    assert "meta" in objs and "google" not in objs


def test_multi_valued_accumulates(engine):
    eng, ns = engine
    eng.ingest_message("I use Postgres.", namespace=ns)
    eng.ingest_message("I use Redis.", namespace=ns)
    current = eng.store.find_current_facts(ns, "user", "uses")
    objs = {f.object.lower() for f in current}
    assert {"postgres", "redis"} <= objs


def test_document_ingest_and_chunk_retrieval(engine):
    eng, ns = engine
    eng.ingest_document("# Runbook\n\nProd runs Postgres 16 in AWS us-east-1.",
                        document_id="rb", namespace=ns)
    results = eng.retrieve("where is prod hosted?", namespace=ns, kinds=("chunk",))
    assert any("us-east-1" in r.text.lower() for r in results)


def test_handoff_and_delete(engine):
    eng, ns = engine
    stored = eng.ingest_message("My name is Grace. I prefer Rust.", namespace=ns)
    h = eng.handoff("who is the user?", namespace=ns)
    assert "grace" in h.text.lower()
    # single-fact delete
    assert eng.forget_fact(stored[0].id) is True
    assert eng.store.get_fact(stored[0].id) is None
