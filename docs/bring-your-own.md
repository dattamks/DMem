# Bring your own infrastructure

DMem is plug-and-play: if you already run Postgres+pgvector, a graph DB, and use
some embedding vendor, you **do not install or create anything new** — you hand
DMem your URLs and keys and it connects. The Docker Compose stack exists only for
people starting from scratch.

Two ways to run, same library:

| You have… | Do this |
|---|---|
| existing Postgres+pgvector, graph DB, embedding vendor | set the URLs/keys below — DMem connects to them |
| nothing yet | `docker compose up` brings up FalkorDB + proxy (see [docker.md](docker.md)) |

## Use your existing databases

DMem never creates a new database and never touches your tables. It creates only
its own **prefixed** objects with `CREATE … IF NOT EXISTS`, so it coexists.

```bash
MEMORY_TIER=pro

# your existing Postgres + pgvector — DMem adds dmem_chunks / dmem_meta only
PGVECTOR_URL=postgresql://user:pass@your-db-host:5432/your_existing_db
# optional: change the prefix so it can never collide, or to run >1 instance
PGVECTOR_TABLE_PREFIX=dmem

# your existing graph DB
GRAPH_DB=neo4j                 # or falkordb
GRAPH_DB_URL=bolt://your-neo4j-host:7687
GRAPH_DB_USER=neo4j
GRAPH_DB_PASSWORD=•••
# GRAPH_DB_DATABASE=dmem        # recommended: a dedicated DB/graph for isolation
```

What DMem writes, and why it's safe to point at a shared DB:

- **pgvector**: tables `dmem_chunks`, `dmem_meta` (prefix configurable). It runs
  `CREATE EXTENSION IF NOT EXISTS vector` — a no-op if you already have pgvector.
  Nothing you own is read or modified.
- **Graph**: labels `Fact`, `Entity`, `Chunk`, `DMemMeta`, and prefixed
  constraints (`dmem_fact_id`, …). FalkorDB isolates in a **named graph**
  (`GRAPH_DB_DATABASE`, default `dmem`); for Neo4j, point `GRAPH_DB_DATABASE` at
  a dedicated database for clean separation.

Don't want two servers? The **consolidated** tier uses one graph DB for both
facts and vectors:

```bash
MEMORY_TIER=consolidated
GRAPH_DB=falkordb
GRAPH_DB_URL=redis://your-falkordb:6379
```

Or the **sqlite** tier for a single-file store with no server at all.

### No graph server, but real graph + Cypher: embedded Kuzu

If you don't run (or don't want) a networked graph server, and Neo4j's GPLv3 /
FalkorDB's SSPL licensing is a concern, use **Kuzu** — an **MIT-licensed embedded
graph database** that runs in-process (like SQLite) with real Cypher and vector
support. One embedded engine holds facts + documents + graph traversal:

```bash
pip install "dmem[kuzu]"
MEMORY_TIER=consolidated
GRAPH_DB=kuzu
GRAPH_DB_URL=./dmem_graph.kz      # a filesystem path, not a network URL
```

Trade-off: embedded means single-process — ideal for an IDE plugin or a single
app embedding DMem, not a shared networked store for many services at scale
(that's what Neo4j/FalkorDB + the pro tier are for).

## Use your embedding vendor

Pick the vendor with `EMBEDDING_PROVIDER` and give it a key — no local model.

```bash
# OpenAI (and any OpenAI-compatible vendor: Qwen/DashScope, vLLM, TEI, Ollama,
# Azure, Together, Fireworks, …) — just point the host URL at the vendor
EMBEDDING_PROVIDER=openai
EMBEDDING_HOST_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-…

# Qwen via DashScope (OpenAI-compatible endpoint)
EMBEDDING_PROVIDER=qwen
EMBEDDING_HOST_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_API_KEY=…

# Google Gemini (native API — no host URL needed)
EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_API_KEY=…

# Cohere (native API)
EMBEDDING_PROVIDER=cohere
EMBEDDING_MODEL=embed-v4.0
EMBEDDING_API_KEY=…
```

Most vendors expose an **OpenAI-compatible** `/embeddings` endpoint — for those,
use `EMBEDDING_PROVIDER=openai` and set `EMBEDDING_HOST_URL` to their base URL.
`google` and `cohere` have native adapters because their API shape differs.

> Changing the embedding model/vendor changes the vector space. DMem stamps the
> embedder signature and warns (or errors) on a mismatch; run `engine.reembed()`
> after an intentional switch. See [architecture.md](architecture.md).

## Same for OCR and the extraction/summary LLM

Both are bring-your-own endpoints too — `OCR_HOST_URL`, and `LLM_HOST_URL` /
`LLM_MODEL` / `LLM_API_KEY`. Unset OCR means text/markdown docs still ingest;
unset LLM falls back to the offline heuristic extractor. See
[`../.env.example`](../.env.example).
