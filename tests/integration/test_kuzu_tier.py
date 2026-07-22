"""Real graph-tier validation via the embedded Kuzu backend — no server, no Docker.

Kuzu is an MIT-licensed embedded graph DB (pip-installable). This exercises the
FULL engine on the consolidated tier (GRAPH_DB=kuzu) in-process: typed facts,
bi-temporal contradiction, multi-valued accumulation, hybrid retrieval with real
Cypher graph traversal, handoff, and GDPR deletes.

    pip install -e ".[dev]" kuzu
    pytest tests/integration/test_kuzu_tier.py -q
"""

import warnings

import pytest

pytest.importorskip("kuzu")

from dmem import Config, DMemEngine, Tier
from dmem.config import EmbeddingConfig, GraphConfig, GraphKind


@pytest.fixture
def engine(tmp_path):
    cfg = Config(
        tier=Tier.CONSOLIDATED,
        embedding=EmbeddingConfig(),  # offline hashing embedder (dim 256)
        graph=GraphConfig(kind=GraphKind.KUZU, url=str(tmp_path / "graph.kz")),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = DMemEngine(cfg)
    yield eng
    eng.close()


def test_ingest_and_retrieve(engine):
    engine.ingest_message("My name is Ada and I work at Analytical Engines.",
                          namespace="u1")
    results = engine.retrieve("where does the user work?", namespace="u1")
    assert any("analytical engines" in r.text.lower() for r in results)


def test_single_valued_contradiction(engine):
    engine.ingest_message("I work at Google.", namespace="u2")
    engine.ingest_message("I work at Meta.", namespace="u2")
    current = engine.store.find_current_facts("u2", "user", "works_at")
    objs = {f.object.lower() for f in current}
    assert "meta" in objs and "google" not in objs
    # old value preserved but closed (bi-temporal)
    assert any(f.object.lower() == "google" and f.valid_to is not None
               for f in engine.store.iter_facts())


def test_multi_valued_accumulates(engine):
    engine.ingest_message("I use Postgres.", namespace="u3")
    engine.ingest_message("I use Redis.", namespace="u3")
    objs = {f.object.lower()
            for f in engine.store.find_current_facts("u3", "user", "uses")}
    assert {"postgres", "redis"} <= objs


def test_graph_traversal_neighbors(engine):
    # real Cypher traversal over Entity/Fact relationships
    engine.ingest_message("My name is Ada. I work at Analytical Engines.",
                          namespace="u4")
    hits = engine.store.graph_neighbors("u4", ["user"], max_hops=1)
    joined = " ".join(h.text.lower() for h in hits)
    assert "analytical engines" in joined or "ada" in joined


def test_document_ingest_and_chunk_retrieval(engine):
    engine.ingest_document("# Runbook\n\nProd runs Postgres 16 in AWS us-east-1.",
                           document_id="rb", namespace="u5")
    results = engine.retrieve("where is prod hosted?", namespace="u5",
                              kinds=("chunk",))
    assert any("us-east-1" in r.text.lower() for r in results)


def test_handoff_and_delete(engine):
    stored = engine.ingest_message("My name is Grace. I prefer Rust.",
                                   namespace="u6")
    h = engine.handoff("who is the user?", namespace="u6")
    assert "grace" in h.text.lower()
    assert engine.forget_fact(stored[0].id) is True
    assert engine.store.get_fact(stored[0].id) is None


def test_namespace_isolation_and_forget(engine):
    engine.ingest_message("My name is Alice.", namespace="a")
    engine.ingest_message("My name is Bob.", namespace="b")
    assert engine.forget("a") >= 1
    assert engine.retrieve("name", namespace="a") == []
    assert any("bob" in r.text.lower()
               for r in engine.retrieve("name", namespace="b"))
