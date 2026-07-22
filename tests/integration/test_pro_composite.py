"""Pro-tier composite against TWO real backends: facts in Kuzu, docs in pgvector.

The production pro tier composes a graph store (facts) with pgvector (documents)
via ProStore. We can't run Neo4j/FalkorDB here, but we CAN validate the composite
routing/merge logic against real engines by using the embedded Kuzu graph for the
fact half and a real bundled Postgres for the document half. Previously ProStore
was only exercised with fake stores.
"""

import pytest

pytest.importorskip("kuzu")
pgserver = pytest.importorskip("pgserver")
pytest.importorskip("psycopg")

from dmem.stores.factory import ProStore
from dmem.stores.kuzu_store import KuzuGraphStore
from dmem.stores.pgvector_store import PgVectorStore
from dmem.types import Chunk, Fact, FactType, Provenance

DIM = 4


@pytest.fixture(scope="module")
def pg_uri(tmp_path_factory):
    server = pgserver.get_server(str(tmp_path_factory.mktemp("compo-pg")))
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


@pytest.fixture
def store(pg_uri, tmp_path):
    graph = KuzuGraphStore(str(tmp_path / "facts.kz"), embed_dim=DIM)
    docs = PgVectorStore(pg_uri, embed_dim=DIM, table_prefix=f"c{abs(hash(str(tmp_path)))%10000}")
    s = ProStore(graph, docs)
    s.initialize()
    yield s
    s.close()


def _fact(subj, pred, obj, emb, ns="t"):
    return Fact(subject=subj, predicate=pred, object=obj, fact_type=FactType.OTHER,
                namespace=ns, embedding=emb,
                provenance=Provenance(source="test", timestamp=1.0))


def test_facts_route_to_graph_docs_to_pgvector(store):
    store.upsert_fact(_fact("user", "works_at", "Meta", [1.0, 0, 0, 0]))
    store.upsert_chunks([Chunk(text="Prod runs in AWS us-east-1", document_id="d",
                               namespace="t", embedding=[0, 1.0, 0, 0])])
    # facts came from the graph half
    facts = store.find_current_facts("t", "user", "works_at")
    assert [f.object for f in facts] == ["Meta"]
    # chunk dedup lives in the pgvector half
    import hashlib
    h = hashlib.sha256(b"Prod runs in AWS us-east-1").hexdigest()
    assert store.chunk_exists("t", h) is True


def test_vector_search_merges_both_halves(store):
    store.upsert_fact(_fact("user", "uses", "Rust", [1.0, 0, 0, 0]))
    store.upsert_chunks([Chunk(text="doc about deployment", document_id="d",
                               namespace="t", embedding=[1.0, 0, 0, 0])])
    hits = store.vector_search("t", [1.0, 0, 0, 0], top_k=5)
    kinds = {h.kind for h in hits}
    assert "fact" in kinds and "chunk" in kinds   # merged from graph + pgvector


def test_graph_neighbors_from_graph_half(store):
    store.upsert_fact(_fact("ada", "knows", "grace", [1.0, 0, 0, 0]))
    hits = store.graph_neighbors("t", ["ada"], max_hops=1)
    assert any("grace" in h.text.lower() for h in hits)


def test_delete_namespace_clears_both(store):
    store.upsert_fact(_fact("x", "is", "y", [1.0, 0, 0, 0], ns="z"))
    store.upsert_chunks([Chunk(text="zdoc", document_id="d", namespace="z",
                               embedding=[1.0, 0, 0, 0])])
    n = store.delete_namespace("z")
    assert n >= 2  # fact + chunk both counted/removed
    assert store.find_current_facts("z", "x", "is") == []
    assert store.vector_search("z", [1.0, 0, 0, 0], top_k=5) == []


def test_meta_via_composite(store):
    store.set_meta("embedding_signature", "sig-1")
    assert store.get_meta("embedding_signature") == "sig-1"
