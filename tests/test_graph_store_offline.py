"""Offline coverage for the graph store (consolidated/pro tiers) without a live
server. Two layers:

1. Driver result-shape normalization — the real Neo4j-vs-FalkorDB bug: Neo4j
   returns Record objects (dict-like, nodes are Mappings); FalkorDB returns
   positional `result_set` rows with a separate header, and its Node has NO
   __getitem__ (only `.properties`). `_CypherDriver.run()` must normalize both.
2. `GraphStore` row-handling logic via an injected fake driver, so fact/vector/
   graph parsing is exercised deterministically.
"""

from collections import deque

import pytest

from dmem.config import GraphConfig, GraphKind
from dmem.stores.graph_store import (_CypherDriver, _count, _fact_text,
                                     _node_to_dict, _node_to_fact, GraphStore)


# --------------------------------------------------------------------------- #
# 1. normalization
# --------------------------------------------------------------------------- #

class _FalkorNode:
    """Mimics falkordb.Node: has .properties, NO __getitem__."""
    def __init__(self, props):
        self.properties = props


def test_node_to_dict_falkordb_node():
    n = _FalkorNode({"id": "fact_1", "subject": "user"})
    d = _node_to_dict(n)
    assert d == {"id": "fact_1", "subject": "user"}
    assert d["id"] == "fact_1"  # dict access works even though Node has none


def test_node_to_dict_neo4j_mapping():
    assert _node_to_dict({"id": "fact_2"}) == {"id": "fact_2"}


def test_node_to_dict_scalar_passthrough():
    assert _node_to_dict(0.87) == 0.87
    assert _node_to_dict(None) is None


class _FakeFalkorResult:
    def __init__(self, header, result_set):
        self.header = header
        self.result_set = result_set


class _FakeFalkorGraph:
    def __init__(self, result):
        self._result = result
        self.calls = []
    def query(self, q, params):
        self.calls.append((q, params))
        return self._result


def _falkor_driver(result):
    d = object.__new__(_CypherDriver)
    d.kind = GraphKind.FALKORDB
    d._graph = _FakeFalkorGraph(result)
    return d


def test_run_normalizes_falkordb_positional_rows():
    # header is [type, name] pairs; result_set rows are positional lists
    result = _FakeFalkorResult(
        header=[[1, b"node"], [2, b"score"]],
        result_set=[[_FalkorNode({"id": "f1", "subject": "user",
                                   "predicate": "uses", "object": "rust"}), 0.91]])
    d = _falkor_driver(result)
    rows = d.run("MATCH ... RETURN node, score")
    assert len(rows) == 1
    assert rows[0]["node"]["id"] == "f1"
    assert rows[0]["score"] == 0.91


def test_run_falkordb_write_without_header():
    result = _FakeFalkorResult(header=[], result_set=[])
    d = _falkor_driver(result)
    assert d.run("CREATE ...") == []


class _FakeNeo4jRecord(dict):
    def keys(self):  # Record-like
        return super().keys()


class _FakeNeo4jResult(list):
    pass


class _FakeSession:
    def __init__(self, records):
        self._records = records
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def run(self, query, **params):
        return _FakeNeo4jResult(self._records)


class _FakeNeo4jDriver:
    def __init__(self, records):
        self._records = records
    def session(self, database=None):
        return _FakeSession(self._records)


def test_run_normalizes_neo4j_records():
    d = object.__new__(_CypherDriver)
    d.kind = GraphKind.NEO4J
    d._database = None
    d._driver = _FakeNeo4jDriver(
        [_FakeNeo4jRecord({"f": {"id": "f9", "subject": "user",
                                 "predicate": "is", "object": "ada"}})])
    rows = d.run("MATCH (f) RETURN f")
    assert rows[0]["f"]["id"] == "f9"


# --------------------------------------------------------------------------- #
# 2. GraphStore logic with an injected fake driver
# --------------------------------------------------------------------------- #

class FakeDriver:
    """Returns canned normalized rows; records queries. Emulates the run()
    contract, not Cypher semantics."""
    kind = GraphKind.NEO4J  # exercises the native vector_query path

    def __init__(self):
        self.run_returns = deque()
        self.vector_returns = deque()
        self.queries = []
        self.vector_calls = []
    def run(self, query, **params):
        self.queries.append((query, params))
        return self.run_returns.popleft() if self.run_returns else []
    def create_vector_index(self, *a, **k):
        pass
    def vector_query(self, label, prop, index_name, k, vec, namespace):
        self.vector_calls.append((label, namespace, k))
        return self.vector_returns.popleft() if self.vector_returns else []
    def close(self):
        pass


def _store(hold_chunks=True):
    fake = FakeDriver()
    gs = GraphStore(GraphConfig(kind=GraphKind.FALKORDB, url="x"),
                    hold_chunks=hold_chunks, driver=fake)
    return gs, fake


FACT_PROPS = {"id": "f1", "namespace": "default", "subject": "user",
              "predicate": "works_at", "object": "Meta", "fact_type": "identity",
              "confidence": 0.9, "valid_from": 1.0, "valid_to": None,
              "recorded_at": 2.0, "supersedes": None, "embedding": None}


def test_find_fact_by_dedup_key_parses_node():
    gs, fake = _store()
    fake.run_returns.append([{"f": dict(FACT_PROPS)}])
    fact = gs.find_fact_by_dedup_key("default", "k")
    assert fact.id == "f1"
    assert fact.object == "Meta"
    assert fact.fact_type.value == "identity"


def test_find_current_facts_parses_list():
    gs, fake = _store()
    fake.run_returns.append([{"f": dict(FACT_PROPS)},
                             {"f": {**FACT_PROPS, "id": "f2", "object": "Google"}}])
    facts = gs.find_current_facts("default", "user", "works_at")
    assert {f.object for f in facts} == {"Meta", "Google"}


def test_vector_search_builds_hits_with_score():
    gs, fake = _store()
    fake.vector_returns.append([{"node": dict(FACT_PROPS), "score": 0.77}])
    hits = gs.vector_search("default", [0.1, 0.2], top_k=5, kinds=("fact",))
    assert len(hits) == 1
    assert hits[0].score == 0.77
    assert hits[0].text == "user works_at Meta"
    assert hits[0].kind == "fact"
    assert fake.vector_calls[0][1] == "default"


def test_graph_neighbors_builds_hits():
    gs, fake = _store()
    fake.run_returns.append([{"f": dict(FACT_PROPS)}])
    hits = gs.graph_neighbors("default", ["user"], max_hops=1)
    assert hits[0].text == "user works_at Meta"
    assert hits[0].hops == 1


def test_delete_fact_reads_count():
    gs, fake = _store()
    fake.run_returns.append([{"c": 1}])
    assert gs.delete_fact("f1") is True
    fake.run_returns.append([{"c": 0}])
    assert gs.delete_fact("nope") is False


def test_delete_facts_requires_filter():
    gs, _ = _store()
    with pytest.raises(ValueError):
        gs.delete_facts("default")


def test_iter_facts_yields_parsed():
    gs, fake = _store()
    fake.run_returns.append([{"f": dict(FACT_PROPS)},
                             {"f": {**FACT_PROPS, "id": "f2"}}])
    facts = list(gs.iter_facts())
    assert [f.id for f in facts] == ["f1", "f2"]


def test_meta_roundtrip_shape():
    gs, fake = _store()
    gs.set_meta("k", "v")
    fake.run_returns.append([{"v": "v"}])
    assert gs.get_meta("k") == "v"


def test_count_helper():
    assert _count([{"c": 3}]) == 3
    assert _count([]) == 0
    assert _count([{"c": None}]) == 0


def test_pro_tier_rejects_chunks():
    from dmem.errors import StoreError
    gs, _ = _store(hold_chunks=False)
    with pytest.raises(StoreError):
        gs.upsert_chunks([])
