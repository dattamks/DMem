# Testing

```bash
pip install -e ".[dev]"
pytest                 # offline suite: SQLite tier + all core logic
```

The offline suite requires **no external services** — the SQLite tier and the
graph-store row logic (via an injected fake driver) run everywhere. Integration
tests that need a live server are **skipped unless configured**.

## Graph tier (consolidated) integration tests

These exercise the full engine against a real Neo4j or FalkorDB. FalkorDB is the
lighter option (a Redis module):

```bash
# FalkorDB
docker run -d -p 6379:6379 falkordb/falkordb
pip install -e ".[falkordb,dev]"
export DMEM_TEST_GRAPH_KIND=falkordb
export DMEM_TEST_GRAPH_URL=redis://localhost:6379
pytest tests/integration -q

# Neo4j 5.11+
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
pip install -e ".[neo4j,dev]"
export DMEM_TEST_GRAPH_KIND=neo4j
export DMEM_TEST_GRAPH_URL=bolt://localhost:7687
export DMEM_TEST_GRAPH_USER=neo4j DMEM_TEST_GRAPH_PASSWORD=password
pytest tests/integration -q
```

Each run uses a unique namespace and tears itself down.

### Known validation gap

Neo4j and FalkorDB diverge on **vector-index DDL and query procedures**
(`db.index.vector.*` vs `db.idx.vector.*`, and different `CREATE VECTOR INDEX`
options). Those dialect branches live in `graph_store._CypherDriver`
(`create_vector_index` / `vector_query`) and are written to documented syntax
but should be validated against your server version — this is exactly what the
integration suite is for. The result-shape normalization (Neo4j Records vs
FalkorDB positional rows + `.properties` nodes) IS covered offline.

## Pro tier (pgvector + graph)

The pro tier composes a graph store (facts) with pgvector (documents). Point a
Postgres+pgvector instance and a graph server at the engine with `MEMORY_TIER=pro`
and the corresponding env vars (see `.env.example`) to smoke-test it; automated
pro-tier integration tests are a follow-up.
