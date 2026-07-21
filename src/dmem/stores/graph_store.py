"""Graph-backed store for the consolidated and pro tiers.

Backends: Neo4j 5.11+ (native vector index) or FalkorDB. They share most Cypher
but diverge in two places that this module isolates in the driver shim:

1. **Result shape.** Neo4j returns ``Record`` objects (``rec["f"]``) whose nodes
   are Mapping-like (``node["id"]``). FalkorDB returns ``result_set`` as
   positional lists whose nodes expose ``.properties`` and have **no**
   ``__getitem__``. ``_CypherDriver.run()`` normalizes BOTH into a uniform
   ``list[dict[str, Any]]`` (column-name -> value; nodes -> plain property
   dicts) so the store logic is backend-neutral.
2. **Vector index DDL + query.** Neo4j uses ``db.index.vector.queryNodes`` and
   ``CREATE VECTOR INDEX``; FalkorDB uses ``db.idx.vector.queryNodes`` and its
   own index DDL. These live in ``create_vector_index`` / ``vector_query``.

- Consolidated tier: this store holds BOTH facts and document chunks (one server
  does both jobs via its native vector index).
- Pro tier: facts/relationships only; documents live in pgvector (see
  ``ProStore``).

Requires a running graph server. The offline suite covers the row-normalization
and store logic with a fake driver; end-to-end coverage lives in
``tests/integration`` and runs only when ``DMEM_TEST_GRAPH_URL`` is set. The
vector-index dialects in particular should be validated against a live server.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..config import GraphConfig, GraphKind
from ..errors import StoreError
from ..types import Chunk, Fact, FactType, Provenance
from .base import GraphHit, VectorHit

FACT_LABEL = "Fact"
CHUNK_LABEL = "Chunk"
FACT_VEC_INDEX = "dmem_fact_vec"
CHUNK_VEC_INDEX = "dmem_chunk_vec"


def _node_to_dict(value: Any) -> Any:
    """Convert a driver node to a plain property dict; pass scalars through.

    Handles Neo4j nodes (Mapping -> dict) and FalkorDB nodes (``.properties``).
    """
    if value is None:
        return None
    props = getattr(value, "properties", None)
    if props is not None:  # FalkorDB Node
        return dict(props)
    # Neo4j Node / Record value is Mapping-like; scalars fall through unchanged.
    try:
        return dict(value)  # Neo4j Node supports dict(); str/float/int do not
    except (TypeError, ValueError):
        return value


def _decode(name: Any) -> str:
    return name.decode() if isinstance(name, (bytes, bytearray)) else str(name)


class _CypherDriver:
    """Backend-neutral shim over Neo4j / FalkorDB. Normalizes result rows."""

    def __init__(self, cfg: GraphConfig):
        self.kind = cfg.kind
        if cfg.kind is GraphKind.NEO4J:
            try:
                from neo4j import GraphDatabase
            except ImportError as e:  # pragma: no cover
                raise StoreError(
                    "Neo4j backend needs the 'neo4j' extra: pip install dmem[neo4j]"
                ) from e
            self._driver = GraphDatabase.driver(
                cfg.url, auth=(cfg.user or "neo4j", cfg.password or "")
            )
            self._database = cfg.database
        elif cfg.kind is GraphKind.FALKORDB:
            try:
                from falkordb import FalkorDB
            except ImportError as e:  # pragma: no cover
                raise StoreError(
                    "FalkorDB backend needs the 'falkordb' extra: "
                    "pip install dmem[falkordb]"
                ) from e
            self._db = FalkorDB.from_url(cfg.url)
            self._graph = self._db.select_graph(cfg.database or "dmem")
        else:  # pragma: no cover
            raise StoreError("GRAPH_DB must be 'neo4j' or 'falkordb'.")

    def run(self, query: str, **params) -> list[dict[str, Any]]:
        """Execute a query and return normalized rows (col name -> value)."""
        if self.kind is GraphKind.NEO4J:
            with self._driver.session(database=self._database) as s:
                result = s.run(query, **params)
                return [
                    {k: _node_to_dict(rec[k]) for k in rec.keys()}
                    for rec in result
                ]
        # FalkorDB: positional result_set + separate header of [type, name] pairs
        result = self._graph.query(query, params)
        rows = getattr(result, "result_set", None) or []
        header = getattr(result, "header", None) or []
        names = [_decode(col[1] if isinstance(col, (list, tuple)) else col)
                 for col in header]
        out: list[dict[str, Any]] = []
        for row in rows:
            if names and len(names) == len(row):
                out.append({names[i]: _node_to_dict(row[i])
                            for i in range(len(row))})
            else:  # no header (writes) — index by position
                out.append({str(i): _node_to_dict(v) for i, v in enumerate(row)})
        return out

    # -- backend-divergent vector operations -------------------------------
    def create_vector_index(self, label: str, prop: str, index_name: str,
                            dim: int) -> None:
        """Create a cosine vector index. Dialect differs per backend."""
        if self.kind is GraphKind.NEO4J:
            self.run(
                f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
                f"FOR (n:{label}) ON n.{prop} "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, "
                f"`vector.similarity_function`: 'cosine'}}}}")
        else:  # FalkorDB — validate DDL against your server version
            try:
                self.run(
                    f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.{prop}) "
                    f"OPTIONS {{dimension: {dim}, similarityFunction: 'cosine'}}")
            except Exception:  # pragma: no cover - older FalkorDB procedure form
                self.run(
                    f"CALL db.idx.vector.createNodeIndex('{label}', '{prop}', "
                    f"{dim}, 'cosine')")

    def vector_query(self, label: str, prop: str, index_name: str, k: int,
                     vec: list[float], namespace: str) -> list[dict[str, Any]]:
        """Return rows of {node, score} for the k nearest vectors in a namespace."""
        if self.kind is GraphKind.NEO4J:
            return self.run(
                f"CALL db.index.vector.queryNodes($idx, $k, $vec) "
                f"YIELD node, score WHERE node.namespace=$ns "
                f"RETURN node, score", idx=index_name, k=k, vec=vec, ns=namespace)
        # FalkorDB vector query (validate against your server version)
        return self.run(
            f"CALL db.idx.vector.queryNodes('{label}', '{prop}', $k, vecf32($vec)) "
            f"YIELD node, score WHERE node.namespace=$ns RETURN node, score",
            k=k, vec=vec, ns=namespace)

    def close(self):
        if self.kind is GraphKind.NEO4J:
            self._driver.close()


class GraphStore:
    """Facts (+ chunks on consolidated tier) in a Cypher graph DB."""

    def __init__(self, cfg: GraphConfig, *, hold_chunks: bool, embed_dim: int = 256,
                 driver: Optional[Any] = None):
        self.tier = "consolidated" if hold_chunks else "pro"
        self._cfg = cfg
        self._hold_chunks = hold_chunks
        self._embed_dim = embed_dim
        # `driver` injectable for offline testing with a fake.
        self._d = driver if driver is not None else _CypherDriver(cfg)

    def initialize(self) -> None:
        try:
            self._d.run(
                "CREATE CONSTRAINT dmem_fact_id IF NOT EXISTS "
                "FOR (f:Fact) REQUIRE f.id IS UNIQUE")
            self._d.run(
                "CREATE CONSTRAINT dmem_entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE (e.namespace, e.name) IS UNIQUE")
            self._d.create_vector_index(FACT_LABEL, "embedding", FACT_VEC_INDEX,
                                        self._embed_dim)
            if self._hold_chunks:
                self._d.create_vector_index(CHUNK_LABEL, "embedding",
                                            CHUNK_VEC_INDEX, self._embed_dim)
        except Exception as e:  # pragma: no cover - server dependent
            raise StoreError(f"Graph initialize failed: {e}") from e

    def close(self) -> None:
        self._d.close()

    # -- facts -------------------------------------------------------------
    def upsert_fact(self, fact: Fact) -> None:
        prov = fact.provenance.to_dict() if fact.provenance else {}
        self._d.run(
            """
            MERGE (s:Entity {namespace:$ns, name:$subject})
            MERGE (o:Entity {namespace:$ns, name:$object})
            CREATE (f:Fact {id:$id, namespace:$ns, subject:$subject,
                predicate:$predicate, object:$object, fact_type:$ftype,
                confidence:$conf, dedup_key:$dedup, embedding:$emb,
                valid_from:$vfrom, valid_to:$vto, recorded_at:$rec,
                supersedes:$sup, provenance:$prov})
            MERGE (s)-[:ASSERTS]->(f)
            MERGE (f)-[:ABOUT]->(o)
            """,
            ns=fact.namespace, id=fact.id, subject=fact.subject,
            predicate=fact.predicate, object=fact.object,
            ftype=fact.fact_type.value, conf=fact.confidence,
            dedup=fact.dedup_key(), emb=fact.embedding, vfrom=fact.valid_from,
            vto=fact.valid_to, rec=fact.recorded_at, sup=fact.supersedes,
            prov=str(prov),
        )

    def find_fact_by_dedup_key(self, namespace: str, dedup_key: str) -> Optional[Fact]:
        rows = self._d.run(
            "MATCH (f:Fact {namespace:$ns, dedup_key:$dk}) "
            "WHERE f.valid_to IS NULL RETURN f LIMIT 1", ns=namespace, dk=dedup_key)
        return _node_to_fact(rows[0]["f"]) if rows else None

    def find_current_facts(self, namespace: str, subject: str,
                           predicate: str) -> list[Fact]:
        rows = self._d.run(
            "MATCH (f:Fact {namespace:$ns}) WHERE toLower(f.subject)=$s "
            "AND toLower(f.predicate)=$p AND f.valid_to IS NULL RETURN f",
            ns=namespace, s=subject.lower(), p=predicate.lower())
        return [_node_to_fact(r["f"]) for r in rows]

    def close_fact(self, fact_id: str, valid_to: float) -> None:
        self._d.run(
            "MATCH (f:Fact {id:$id}) WHERE f.valid_to IS NULL SET f.valid_to=$vto",
            id=fact_id, vto=valid_to)

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        rows = self._d.run("MATCH (f:Fact {id:$id}) RETURN f", id=fact_id)
        return _node_to_fact(rows[0]["f"]) if rows else None

    def delete_fact(self, fact_id: str) -> bool:
        rows = self._d.run(
            "MATCH (f:Fact {id:$id}) DETACH DELETE f RETURN count(f) AS c",
            id=fact_id)
        return _count(rows) > 0

    def delete_facts(self, namespace: str, *, subject=None, predicate=None,
                     object=None) -> int:
        if not any([subject, predicate, object]):
            raise ValueError("delete_facts needs at least one filter.")
        clauses = ["f.namespace=$ns"]
        params: dict = {"ns": namespace}
        if subject:
            clauses.append("toLower(f.subject)=$subject")
            params["subject"] = subject.strip().lower()
        if predicate:
            clauses.append("toLower(f.predicate)=$predicate")
            params["predicate"] = predicate.strip().lower()
        if object:
            clauses.append("toLower(f.object)=$object")
            params["object"] = object.strip().lower()
        rows = self._d.run(
            f"MATCH (f:Fact) WHERE {' AND '.join(clauses)} "
            f"DETACH DELETE f RETURN count(f) AS c", **params)
        return _count(rows)

    # -- documents (consolidated tier only) --------------------------------
    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        if not self._hold_chunks:
            raise StoreError("This graph store does not hold chunks (pro tier "
                             "uses pgvector for documents).")
        for c in chunks:
            self._d.run(
                """CREATE (c:Chunk {id:$id, namespace:$ns, document_id:$doc,
                    concept:$concept, section:$section, ordinal:$ord, text:$text,
                    content_hash:$hash, embedding:$emb})""",
                id=c.id, ns=c.namespace, doc=c.document_id, concept=c.concept,
                section=c.section, ord=c.ordinal, text=c.text,
                hash=_hash(c.text), emb=c.embedding)

    def chunk_exists(self, namespace: str, content_hash: str) -> bool:
        if not self._hold_chunks:
            return False
        rows = self._d.run(
            "MATCH (c:Chunk {namespace:$ns, content_hash:$h}) RETURN c LIMIT 1",
            ns=namespace, h=content_hash)
        return bool(rows)

    # -- retrieval ---------------------------------------------------------
    def vector_search(self, namespace, query_embedding, top_k,
                      kinds=("fact", "chunk")) -> list[VectorHit]:
        hits: list[VectorHit] = []
        if "fact" in kinds:
            for r in self._d.vector_query(FACT_LABEL, "embedding", FACT_VEC_INDEX,
                                          top_k, query_embedding, namespace):
                n = r["node"]
                hits.append(VectorHit(id=n["id"], text=_fact_text(n),
                                      score=float(r.get("score", 0.0)),
                                      kind="fact", payload=n))
        if "chunk" in kinds and self._hold_chunks:
            for r in self._d.vector_query(CHUNK_LABEL, "embedding", CHUNK_VEC_INDEX,
                                          top_k, query_embedding, namespace):
                n = r["node"]
                hits.append(VectorHit(id=n["id"], text=n.get("text", ""),
                                      score=float(r.get("score", 0.0)),
                                      kind="chunk", payload=n))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def keyword_search(self, namespace, query, top_k,
                       kinds=("fact", "chunk")) -> list[VectorHit]:
        import re
        toks = list({t for t in re.findall(r"[a-z0-9]+", query.lower())})[:8]
        if not toks:
            return []
        hits: list[VectorHit] = []
        if "fact" in kinds:
            rows = self._d.run(
                "MATCH (f:Fact {namespace:$ns}) WHERE any(t IN $toks WHERE "
                "toLower(f.subject+' '+f.predicate+' '+f.object) CONTAINS t) "
                "RETURN f LIMIT $k", ns=namespace, toks=toks, k=top_k)
            for r in rows:
                n = r["f"]
                hits.append(VectorHit(id=n["id"], text=_fact_text(n), score=0.5,
                                      kind="fact", payload=n))
        if "chunk" in kinds and self._hold_chunks:
            rows = self._d.run(
                "MATCH (c:Chunk {namespace:$ns}) WHERE any(t IN $toks WHERE "
                "toLower(c.text) CONTAINS t) RETURN c LIMIT $k",
                ns=namespace, toks=toks, k=top_k)
            for r in rows:
                n = r["c"]
                hits.append(VectorHit(id=n["id"], text=n.get("text", ""),
                                      score=0.5, kind="chunk", payload=n))
        return hits[:top_k]

    def graph_neighbors(self, namespace, seeds, max_hops=1, limit=20) -> list[GraphHit]:
        if not seeds:
            return []
        rows = self._d.run(
            f"""MATCH (e:Entity {{namespace:$ns}})-[*1..{int(max_hops)}]-(f:Fact)
                WHERE toLower(e.name) IN $seeds AND f.valid_to IS NULL
                RETURN DISTINCT f LIMIT $lim""",
            ns=namespace, seeds=[s.lower() for s in seeds], lim=limit)
        return [GraphHit(id=r["f"]["id"], text=_fact_text(r["f"]), hops=1,
                         payload=r["f"]) for r in rows]

    # -- meta --------------------------------------------------------------
    def get_meta(self, key: str) -> Optional[str]:
        rows = self._d.run("MATCH (m:DMemMeta {key:$k}) RETURN m.value AS v", k=key)
        return rows[0].get("v") if rows else None

    def set_meta(self, key: str, value: str) -> None:
        self._d.run("MERGE (m:DMemMeta {key:$k}) SET m.value=$v", k=key, v=value)

    def iter_facts(self):
        for r in self._d.run("MATCH (f:Fact) RETURN f"):
            yield _node_to_fact(r["f"])

    def iter_chunks(self):
        if not self._hold_chunks:
            return
        for r in self._d.run("MATCH (c:Chunk) RETURN c"):
            n = r["c"]
            yield Chunk(text=n.get("text", ""), document_id=n.get("document_id", ""),
                        concept=n.get("concept"), section=n.get("section"),
                        ordinal=n.get("ordinal", 0), id=n["id"],
                        namespace=n["namespace"], embedding=n.get("embedding"))

    def delete_namespace(self, namespace: str) -> int:
        rows = self._d.run(
            "MATCH (n {namespace:$ns}) DETACH DELETE n RETURN count(n) AS c",
            ns=namespace)
        return _count(rows)


def _fact_text(n: dict) -> str:
    return f"{n.get('subject','')} {n.get('predicate','')} {n.get('object','')}".strip()


def _count(rows: list[dict]) -> int:
    if not rows:
        return 0
    val = rows[0].get("c", 0)
    try:
        return int(val)
    except (TypeError, ValueError):  # pragma: no cover
        return 0


def _node_to_fact(n: dict) -> Fact:
    return Fact(
        subject=n["subject"], predicate=n["predicate"], object=n["object"],
        fact_type=FactType.coerce(n.get("fact_type")),
        provenance=None, id=n["id"], namespace=n["namespace"],
        confidence=n.get("confidence", 1.0), embedding=n.get("embedding"),
        valid_from=n.get("valid_from"), valid_to=n.get("valid_to"),
        recorded_at=n.get("recorded_at"), supersedes=n.get("supersedes"))


def _hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
