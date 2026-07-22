"""Kuzu embedded graph backend (MIT-licensed, in-process, no server).

Kuzu is an embedded graph database — like SQLite for graphs — that speaks Cypher
and installs via pip. Unlike Neo4j (GPLv3) and FalkorDB (SSPL), it is MIT, and it
needs no server: the whole store lives in a local directory. It holds BOTH facts
and document chunks (the "consolidated" role), so a single embedded engine gives
real Cypher graph traversal plus vector/keyword retrieval.

Config: ``MEMORY_TIER=consolidated GRAPH_DB=kuzu GRAPH_DB_URL=/path/to/graph.kz``
(``GRAPH_DB_URL`` is a filesystem path here, not a network URL).

Vectors use brute-force cosine in Python (as the SQLite tier does) — fine for the
single-process scale this backend targets. Graph traversal is native Cypher.
"""

from __future__ import annotations

import json
import math
import re
import threading
from typing import Optional, Sequence

from ..errors import StoreError
from ..types import Chunk, Fact, FactType, Provenance
from .base import GraphHit, VectorHit

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_FACT_COLS = (
    "f.id AS id, f.namespace AS namespace, f.subject AS subject, "
    "f.predicate AS predicate, f.object AS object, f.fact_type AS fact_type, "
    "f.confidence AS confidence, f.emb AS emb, f.valid_from AS valid_from, "
    "f.valid_to AS valid_to, f.recorded_at AS recorded_at, "
    "f.supersedes AS supersedes, f.provenance AS provenance")


def _cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _token_overlap(query_tokens: set, text: str) -> float:
    ttoks = _TOKEN_RE.findall(text.lower())
    if not ttoks:
        return 0.0
    tset = set(ttoks)
    overlap = len(query_tokens & tset)
    return overlap / (math.sqrt(len(tset)) + 1.0) if overlap else 0.0


class KuzuGraphStore:
    tier = "consolidated"

    def __init__(self, path: str, embed_dim: int = 256):
        try:
            import kuzu
        except ImportError as e:  # pragma: no cover
            raise StoreError(
                "The Kuzu backend needs the 'kuzu' extra: pip install dmem[kuzu]"
            ) from e
        self._kuzu = kuzu
        self._path = path
        self._embed_dim = embed_dim
        self._db = kuzu.Database(path)
        self._conn = kuzu.Connection(self._db)
        self._lock = threading.RLock()

    # -- lifecycle ---------------------------------------------------------
    def initialize(self) -> None:
        ddl = [
            "CREATE NODE TABLE IF NOT EXISTS Fact("
            "id STRING, namespace STRING, subject STRING, predicate STRING, "
            "object STRING, fact_type STRING, confidence DOUBLE, "
            "dedup_key STRING, emb DOUBLE[], valid_from DOUBLE, valid_to DOUBLE, "
            "recorded_at DOUBLE, supersedes STRING, provenance STRING, "
            "PRIMARY KEY(id))",
            "CREATE NODE TABLE IF NOT EXISTS Chunk("
            "id STRING, namespace STRING, document_id STRING, concept STRING, "
            "section STRING, ordinal INT64, text STRING, content_hash STRING, "
            "emb DOUBLE[], PRIMARY KEY(id))",
            "CREATE NODE TABLE IF NOT EXISTS Entity("
            "key STRING, namespace STRING, name STRING, PRIMARY KEY(key))",
            "CREATE NODE TABLE IF NOT EXISTS Meta(key STRING, value STRING, "
            "PRIMARY KEY(key))",
            "CREATE REL TABLE IF NOT EXISTS ASSERTS(FROM Entity TO Fact)",
            "CREATE REL TABLE IF NOT EXISTS ABOUT(FROM Fact TO Entity)",
        ]
        with self._lock:
            for stmt in ddl:
                self._conn.execute(stmt)

    def close(self) -> None:
        # Kuzu closes with the process; drop references.
        self._conn = None
        self._db = None

    # -- helpers -----------------------------------------------------------
    def _rows(self, cypher: str, **params) -> list[dict]:
        with self._lock:
            res = self._conn.execute(cypher, parameters=params or {})
        cols = [c.split(".")[-1] if "." not in c.split(" AS ")[-1] else c
                for c in res.get_column_names()]
        out = []
        while res.has_next():
            out.append(dict(zip(res.get_column_names(), res.get_next())))
        return out

    def _exec(self, cypher: str, **params) -> None:
        with self._lock:
            self._conn.execute(cypher, parameters=params or {})

    def _entity_key(self, namespace: str, name: str) -> str:
        return f"{namespace}\x1f{name}"

    # -- facts -------------------------------------------------------------
    def upsert_fact(self, fact: Fact) -> None:
        # Kuzu binds parameters strictly: each query gets ONLY the params it
        # references, so we pass per-statement subsets.
        prov = json.dumps(fact.provenance.to_dict()) if fact.provenance else None
        skey = self._entity_key(fact.namespace, fact.subject)
        okey = self._entity_key(fact.namespace, fact.object)
        emb = [float(x) for x in (fact.embedding or [])]
        with self._lock:
            self._conn.execute(
                "MERGE (f:Fact {id:$id}) SET f.namespace=$ns, f.subject=$subject, "
                "f.predicate=$predicate, f.object=$object, f.fact_type=$ftype, "
                "f.confidence=$conf, f.dedup_key=$dedup, f.emb=$emb, "
                "f.valid_from=$vfrom, f.valid_to=$vto, f.recorded_at=$rec, "
                "f.supersedes=$sup, f.provenance=$prov",
                parameters=dict(
                    id=fact.id, ns=fact.namespace, subject=fact.subject,
                    predicate=fact.predicate, object=fact.object,
                    ftype=fact.fact_type.value, conf=float(fact.confidence),
                    dedup=fact.dedup_key(), emb=emb, vfrom=float(fact.valid_from),
                    vto=fact.valid_to, rec=float(fact.recorded_at),
                    sup=fact.supersedes, prov=prov))
            self._conn.execute(
                "MERGE (e:Entity {key:$skey}) SET e.namespace=$ns, e.name=$name",
                parameters=dict(skey=skey, ns=fact.namespace, name=fact.subject))
            self._conn.execute(
                "MERGE (e:Entity {key:$okey}) SET e.namespace=$ns, e.name=$name",
                parameters=dict(okey=okey, ns=fact.namespace, name=fact.object))
            self._conn.execute(
                "MATCH (e:Entity {key:$skey}), (f:Fact {id:$id}) "
                "MERGE (e)-[:ASSERTS]->(f)", parameters=dict(skey=skey, id=fact.id))
            self._conn.execute(
                "MATCH (f:Fact {id:$id}), (e:Entity {key:$okey}) "
                "MERGE (f)-[:ABOUT]->(e)", parameters=dict(id=fact.id, okey=okey))

    def _row_to_fact(self, d: dict) -> Fact:
        prov = json.loads(d["provenance"]) if d.get("provenance") else None
        emb = d.get("emb")
        return Fact(
            subject=d["subject"], predicate=d["predicate"], object=d["object"],
            fact_type=FactType.coerce(d.get("fact_type")),
            provenance=Provenance.from_dict(prov) if prov else None,
            id=d["id"], namespace=d["namespace"],
            confidence=d.get("confidence", 1.0),
            embedding=list(emb) if emb else None,
            valid_from=d.get("valid_from"), valid_to=d.get("valid_to"),
            recorded_at=d.get("recorded_at"), supersedes=d.get("supersedes"))

    def find_fact_by_dedup_key(self, namespace: str, dedup_key: str) -> Optional[Fact]:
        rows = self._rows(
            f"MATCH (f:Fact) WHERE f.namespace=$ns AND f.dedup_key=$dk "
            f"AND f.valid_to IS NULL RETURN {_FACT_COLS} LIMIT 1",
            ns=namespace, dk=dedup_key)
        return self._row_to_fact(rows[0]) if rows else None

    def find_current_facts(self, namespace: str, subject: str,
                           predicate: str) -> list[Fact]:
        rows = self._rows(
            f"MATCH (f:Fact) WHERE f.namespace=$ns AND lower(f.subject)=$s "
            f"AND lower(f.predicate)=$p AND f.valid_to IS NULL RETURN {_FACT_COLS}",
            ns=namespace, s=subject.strip().lower(), p=predicate.strip().lower())
        return [self._row_to_fact(r) for r in rows]

    def close_fact(self, fact_id: str, valid_to: float) -> None:
        self._exec("MATCH (f:Fact {id:$id}) WHERE f.valid_to IS NULL "
                   "SET f.valid_to=$vto", id=fact_id, vto=float(valid_to))

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        rows = self._rows(f"MATCH (f:Fact {{id:$id}}) RETURN {_FACT_COLS}",
                          id=fact_id)
        return self._row_to_fact(rows[0]) if rows else None

    def delete_fact(self, fact_id: str) -> bool:
        before = self._rows("MATCH (f:Fact {id:$id}) RETURN f.id AS id", id=fact_id)
        if not before:
            return False
        self._exec("MATCH (f:Fact {id:$id}) DETACH DELETE f", id=fact_id)
        return True

    def delete_facts(self, namespace: str, *, subject=None, predicate=None,
                     object=None) -> int:
        if not any([subject, predicate, object]):
            raise ValueError("delete_facts needs at least one filter.")
        clauses = ["f.namespace=$ns"]
        p = {"ns": namespace}
        if subject:
            clauses.append("lower(f.subject)=$subject"); p["subject"] = subject.strip().lower()
        if predicate:
            clauses.append("lower(f.predicate)=$predicate"); p["predicate"] = predicate.strip().lower()
        if object:
            clauses.append("lower(f.object)=$object"); p["object"] = object.strip().lower()
        where = " AND ".join(clauses)
        rows = self._rows(f"MATCH (f:Fact) WHERE {where} RETURN count(f) AS c", **p)
        n = int(rows[0]["c"]) if rows else 0
        if n:
            self._exec(f"MATCH (f:Fact) WHERE {where} DETACH DELETE f", **p)
        return n

    # -- documents ---------------------------------------------------------
    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            self._exec(
                "MERGE (c:Chunk {id:$id}) SET c.namespace=$ns, c.document_id=$doc, "
                "c.concept=$concept, c.section=$section, c.ordinal=$ord, "
                "c.text=$text, c.content_hash=$hash, c.emb=$emb",
                id=c.id, ns=c.namespace, doc=c.document_id, concept=c.concept,
                section=c.section, ord=int(c.ordinal), text=c.text,
                hash=_hash(c.text), emb=[float(x) for x in (c.embedding or [])])

    def chunk_exists(self, namespace: str, content_hash: str) -> bool:
        rows = self._rows(
            "MATCH (c:Chunk) WHERE c.namespace=$ns AND c.content_hash=$h "
            "RETURN c.id AS id LIMIT 1", ns=namespace, h=content_hash)
        return bool(rows)

    # -- retrieval ---------------------------------------------------------
    def vector_search(self, namespace, query_embedding, top_k,
                      kinds=("fact", "chunk")) -> list[VectorHit]:
        hits: list[VectorHit] = []
        if "fact" in kinds:
            for d in self._rows(f"MATCH (f:Fact) WHERE f.namespace=$ns RETURN {_FACT_COLS}",
                                ns=namespace):
                if not d.get("emb"):
                    continue
                text = f"{d['subject']} {d['predicate']} {d['object']}".strip()
                hits.append(VectorHit(id=d["id"], text=text,
                                      score=_cosine(query_embedding, d["emb"]),
                                      kind="fact", payload=d))
        if "chunk" in kinds:
            for d in self._rows(
                "MATCH (c:Chunk) WHERE c.namespace=$ns RETURN c.id AS id, "
                "c.text AS text, c.emb AS emb, c.document_id AS document_id, "
                "c.concept AS concept, c.section AS section", ns=namespace):
                if not d.get("emb"):
                    continue
                hits.append(VectorHit(id=d["id"], text=d["text"],
                                      score=_cosine(query_embedding, d["emb"]),
                                      kind="chunk", payload=d))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def keyword_search(self, namespace, query, top_k,
                       kinds=("fact", "chunk")) -> list[VectorHit]:
        toks = set(_TOKEN_RE.findall(query.lower()))
        if not toks:
            return []
        hits: list[VectorHit] = []
        if "fact" in kinds:
            for d in self._rows(f"MATCH (f:Fact) WHERE f.namespace=$ns RETURN {_FACT_COLS}",
                                ns=namespace):
                text = f"{d['subject']} {d['predicate']} {d['object']}".strip()
                s = _token_overlap(toks, text)
                if s > 0:
                    hits.append(VectorHit(id=d["id"], text=text, score=s,
                                          kind="fact", payload=d))
        if "chunk" in kinds:
            for d in self._rows(
                "MATCH (c:Chunk) WHERE c.namespace=$ns RETURN c.id AS id, "
                "c.text AS text, c.document_id AS document_id, "
                "c.concept AS concept, c.section AS section", ns=namespace):
                s = _token_overlap(toks, d["text"])
                if s > 0:
                    hits.append(VectorHit(id=d["id"], text=d["text"], score=s,
                                          kind="chunk", payload=d))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def graph_neighbors(self, namespace, seeds, max_hops=1, limit=20) -> list[GraphHit]:
        if not seeds:
            return []
        names = [s.strip().lower() for s in seeds if s]
        rows = self._rows(
            f"MATCH (e:Entity)-[*1..{int(max_hops)}]-(f:Fact) "
            f"WHERE e.namespace=$ns AND lower(e.name) IN $names "
            f"AND f.valid_to IS NULL RETURN DISTINCT {_FACT_COLS} LIMIT {int(limit)}",
            ns=namespace, names=names)
        return [GraphHit(id=d["id"],
                         text=f"{d['subject']} {d['predicate']} {d['object']}".strip(),
                         hops=1, payload=d) for d in rows]

    # -- meta --------------------------------------------------------------
    def get_meta(self, key: str) -> Optional[str]:
        rows = self._rows("MATCH (m:Meta {key:$k}) RETURN m.value AS value", k=key)
        return rows[0]["value"] if rows else None

    def set_meta(self, key: str, value: str) -> None:
        self._exec("MERGE (m:Meta {key:$k}) SET m.value=$v", k=key, v=value)

    def iter_facts(self):
        for d in self._rows(f"MATCH (f:Fact) RETURN {_FACT_COLS}"):
            yield self._row_to_fact(d)

    def iter_chunks(self):
        for d in self._rows(
            "MATCH (c:Chunk) RETURN c.id AS id, c.namespace AS namespace, "
            "c.document_id AS document_id, c.concept AS concept, "
            "c.section AS section, c.ordinal AS ordinal, c.text AS text, "
            "c.emb AS emb"):
            yield Chunk(text=d["text"], document_id=d["document_id"],
                        concept=d.get("concept"), section=d.get("section"),
                        ordinal=d.get("ordinal", 0), id=d["id"],
                        namespace=d["namespace"],
                        embedding=list(d["emb"]) if d.get("emb") else None)

    def delete_namespace(self, namespace: str) -> int:
        # Kuzu has no label-disjunction predicate; delete per label.
        total = 0
        for label in ("Fact", "Chunk", "Entity"):
            rows = self._rows(
                f"MATCH (n:{label}) WHERE n.namespace=$ns RETURN count(n) AS c",
                ns=namespace)
            c = int(rows[0]["c"]) if rows else 0
            if label in ("Fact", "Chunk"):
                total += c  # count facts+chunks as the deleted-record tally
            if c:
                self._exec(f"MATCH (n:{label}) WHERE n.namespace=$ns "
                           f"DETACH DELETE n", ns=namespace)
        return total


def _hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
