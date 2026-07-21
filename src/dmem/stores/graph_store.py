"""Graph-backed store for the consolidated and pro tiers.

Backends: Neo4j 5.11+ (native vector index) or FalkorDB. Both speak a Cypher
dialect, so one implementation covers them with a thin driver shim.

- Consolidated tier: this store holds BOTH facts and document chunks, using the
  graph DB's native vector index for similarity — one server does both jobs.
- Pro tier: this store holds facts/relationships only; documents live in
  pgvector (see `ProStore`). This is where real multi-hop `graph_neighbors`
  traversal and bi-temporal relationship modeling live.

Requires a running graph server; not exercised by the offline test suite. Kept
behind lazy imports so the core package installs without these drivers.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..config import GraphConfig, GraphKind
from ..errors import StoreError
from ..types import Chunk, Fact, FactType, Provenance
from .base import GraphHit, VectorHit


class _CypherDriver:
    """Minimal shim over Neo4j / FalkorDB so the store code is backend-neutral."""

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

    def run(self, query: str, **params):
        if self.kind is GraphKind.NEO4J:
            with self._driver.session(database=self._database) as s:
                return list(s.run(query, **params))
        result = self._graph.query(query, params)
        return result.result_set

    def close(self):
        if self.kind is GraphKind.NEO4J:
            self._driver.close()


class GraphStore:
    """Facts (+ chunks on consolidated tier) in a Cypher graph DB."""

    def __init__(self, cfg: GraphConfig, *, hold_chunks: bool, embed_dim: int = 256):
        self.tier = "consolidated" if hold_chunks else "pro"
        self._cfg = cfg
        self._hold_chunks = hold_chunks
        self._embed_dim = embed_dim
        self._d = _CypherDriver(cfg)

    def initialize(self) -> None:
        # Uniqueness + vector index. Guarded so re-init is idempotent.
        try:
            self._d.run(
                "CREATE CONSTRAINT dmem_fact_id IF NOT EXISTS "
                "FOR (f:Fact) REQUIRE f.id IS UNIQUE"
            )
            self._d.run(
                "CREATE CONSTRAINT dmem_entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE (e.namespace, e.name) IS UNIQUE"
            )
            self._d.run(
                f"CREATE VECTOR INDEX dmem_fact_vec IF NOT EXISTS "
                f"FOR (f:Fact) ON f.embedding "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {self._embed_dim}, "
                f"`vector.similarity_function`: 'cosine'}}}}"
            )
            if self._hold_chunks:
                self._d.run(
                    f"CREATE VECTOR INDEX dmem_chunk_vec IF NOT EXISTS "
                    f"FOR (c:Chunk) ON c.embedding "
                    f"OPTIONS {{indexConfig: {{`vector.dimensions`: {self._embed_dim}, "
                    f"`vector.similarity_function`: 'cosine'}}}}"
                )
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
        return _row_to_fact(rows[0]) if rows else None

    def find_current_facts(self, namespace: str, subject: str,
                           predicate: str) -> list[Fact]:
        rows = self._d.run(
            "MATCH (f:Fact {namespace:$ns}) WHERE toLower(f.subject)=$s "
            "AND toLower(f.predicate)=$p AND f.valid_to IS NULL RETURN f",
            ns=namespace, s=subject.lower(), p=predicate.lower())
        return [_row_to_fact(r) for r in rows]

    def close_fact(self, fact_id: str, valid_to: float) -> None:
        self._d.run(
            "MATCH (f:Fact {id:$id}) WHERE f.valid_to IS NULL SET f.valid_to=$vto",
            id=fact_id, vto=valid_to)

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        rows = self._d.run("MATCH (f:Fact {id:$id}) RETURN f", id=fact_id)
        return _row_to_fact(rows[0]) if rows else None

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
            rows = self._d.run(
                "CALL db.index.vector.queryNodes('dmem_fact_vec', $k, $q) "
                "YIELD node, score WHERE node.namespace=$ns RETURN node, score",
                k=top_k, q=query_embedding, ns=namespace)
            for r in rows:
                n = r["node"]
                hits.append(VectorHit(id=n["id"], text=_fact_text(n),
                                      score=float(r["score"]), kind="fact",
                                      payload=dict(n)))
        if "chunk" in kinds and self._hold_chunks:
            rows = self._d.run(
                "CALL db.index.vector.queryNodes('dmem_chunk_vec', $k, $q) "
                "YIELD node, score WHERE node.namespace=$ns RETURN node, score",
                k=top_k, q=query_embedding, ns=namespace)
            for r in rows:
                n = r["node"]
                hits.append(VectorHit(id=n["id"], text=n["text"],
                                      score=float(r["score"]), kind="chunk",
                                      payload=dict(n)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def keyword_search(self, namespace, query, top_k,
                       kinds=("fact", "chunk")) -> list[VectorHit]:
        # CONTAINS-based fallback; a full-text index is a future optimization.
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
                                      kind="fact", payload=dict(n)))
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
                         payload=dict(r["f"])) for r in rows]

    def delete_namespace(self, namespace: str) -> int:
        rows = self._d.run(
            "MATCH (n {namespace:$ns}) DETACH DELETE n RETURN count(n) AS c",
            ns=namespace)
        try:
            return int(rows[0]["c"]) if rows else 0
        except Exception:  # pragma: no cover
            return 0


def _fact_text(n) -> str:
    return f"{n['subject']} {n['predicate']} {n['object']}".strip()


def _row_to_fact(row) -> Fact:
    n = row["f"] if "f" in _keys(row) else row["node"] if "node" in _keys(row) else row[0]
    prov = None
    return Fact(
        subject=n["subject"], predicate=n["predicate"], object=n["object"],
        fact_type=FactType.coerce(n.get("fact_type")),
        provenance=prov, id=n["id"], namespace=n["namespace"],
        confidence=n.get("confidence", 1.0), embedding=n.get("embedding"),
        valid_from=n.get("valid_from"), valid_to=n.get("valid_to"),
        recorded_at=n.get("recorded_at"), supersedes=n.get("supersedes"))


def _keys(row):
    try:
        return set(row.keys())
    except Exception:  # pragma: no cover
        return set()


def _hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
