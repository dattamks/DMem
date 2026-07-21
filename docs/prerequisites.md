# Prerequisites (bring your own)

DMem is **bring-your-own-infrastructure**. It does **not** install, provision, or
bootstrap anything. You supply the components; DMem connects to them. This is the
same for all three ways you consume DMem — embedded in an IDE, linked as an MCP
server, or integrated into your application.

The production tier (**`pro`**, the default) has **three mandatory
prerequisites**. If any is missing, DMem stops at startup with an actionable
message — it never silently downgrades.

| Prerequisite | Env vars | Notes |
|---|---|---|
| **Graph database** | `GRAPH_DB` (`neo4j`\|`falkordb`), `GRAPH_DB_URL`, `GRAPH_DB_USER`, `GRAPH_DB_PASSWORD` | Your existing graph DB. Pick the type you run. |
| **Postgres + pgvector** | `PGVECTOR_URL` | Your existing Postgres. DMem **checks that the `vector` extension is enabled** and tells you exactly how to fix it if not. |
| **Embedding endpoint** | `EMBEDDING_PROVIDER` + (`EMBEDDING_HOST_URL` / `EMBEDDING_API_KEY`) | OpenAI-compatible (Qwen, vLLM, Azure, …), `google`, or `cohere`. |

Optional (DMem degrades gracefully without them): `LLM_HOST_URL` (better fact
extraction) and `OCR_HOST_URL` (binary-document ingestion).

## Check your setup before running: `dmem doctor`

```bash
pip install dmem
dmem-doctor
```

It inspects your environment and reports READY / NOT READY per prerequisite —
including a **live probe that connects to your Postgres and verifies pgvector is
enabled** — with the exact fix for each gap. Exit code 0 when ready, 1 otherwise
(usable in CI or an IDE setup check).

```
DMem preflight (dmem doctor)
────────────────────────────────────────────────────
  ✓ Tier                   pro
  ✓ Graph database         neo4j @ bolt://…:7687 — reachable
  ✗ Postgres + pgvector    …@db:5432/app — pgvector NOT enabled: run CREATE EXTENSION vector;
  ✓ Embedding endpoint     openai @ https://api.openai.com/v1
────────────────────────────────────────────────────
Verdict: NOT READY — 1 prerequisite(s) missing.
```

## Minimum config

```bash
MEMORY_TIER=pro

GRAPH_DB=neo4j
GRAPH_DB_URL=bolt://your-neo4j:7687
GRAPH_DB_USER=neo4j
GRAPH_DB_PASSWORD=•••

PGVECTOR_URL=postgresql://user:pass@your-postgres:5432/your_db

EMBEDDING_PROVIDER=openai
EMBEDDING_HOST_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-…
```

Install the client extras for your backends:

```bash
pip install "dmem[neo4j,pgvector,http]"      # or [falkordb,pgvector,http]
```

## "I don't have these yet"

Then they're prerequisites you need to provision **yourself** — DMem won't do it
for you. Point it at a managed Postgres+pgvector and a managed/self-run graph DB,
or (for local development only) see [docker.md](docker.md) for a Compose stack.

## Enabling pgvector on your Postgres

DMem tries to enable it automatically (`CREATE EXTENSION IF NOT EXISTS vector`).
If the DMem role lacks privileges, a superuser runs once:

```sql
CREATE EXTENSION vector;
```

If the server doesn't have pgvector installed at all, install it first:
<https://github.com/pgvector/pgvector>.

## Dev/test tiers (not for production)

`MEMORY_TIER=sqlite` (single file, no servers) and `MEMORY_TIER=consolidated`
(one graph DB doing both jobs) exist **only** for local development and the test
suite. They are not the supported production configuration and `dmem doctor`
flags them as such.
