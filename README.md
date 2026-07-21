# DMem

**Pluggable, self-hostable AI memory & context library.** Persistent memory +
document RAG + model-switch continuity — distributed as a library you drop into
*any* ecosystem, not a hosted service. One core engine; thin adapters for the
Python SDK, an MCP server, and (planned) an IDE extension.

> Open-source realization of the memory / token-efficiency layer from the
> "AI Gateway" spec. The proxy, auth, and billing pieces of that spec are **out
> of scope** here — DMem is the self-contained memory library only.

## Why

Switching models mid-conversation normally means resending the whole prior
context; long conversations and repeated documents cost the same per token
forever. DMem persists **typed facts** and **document knowledge** across
sessions and model switches, and hands a new model a **compact, token-budgeted
summary** instead of the full transcript.

## Install

```bash
pip install dmem                # core: zero third-party deps, SQLite tier
pip install "dmem[http]"        # + remote embedding / OCR / LLM endpoints
pip install "dmem[mcp]"         # + MCP server adapter
pip install "dmem[neo4j]"       # consolidated / pro graph backend
pip install "dmem[pgvector]"    # pro-tier document store
pip install "dmem[rerank]"      # cross-encoder reranking
```

## Quick start (zero setup)

```python
from dmem import DMemEngine

mem = DMemEngine()   # MEMORY_TIER=sqlite by default; no server required

mem.ingest_message("My name is Ada and I prefer Rust. I work at Analytical Engines.")
mem.ingest_document("# Runbook\n\nProd runs Postgres 16 in AWS us-east-1.",
                    document_id="runbook")

handoff = mem.handoff("who is the user and where is prod?")
print(handoff.text)
# Context handoff (compact):
# - user has_name Ada
# - user works_at Analytical Engines
# - user prefers Rust
# - [Runbook] Prod runs Postgres 16 in AWS us-east-1.
```

Out of the box this uses an **offline hashing embedder** (dev quality, warns on
use). Point `EMBEDDING_HOST_URL` at any OpenAI-compatible `/embeddings` endpoint
for production quality — nothing about a provider is hardcoded.

## Tiers

Pick explicitly via `MEMORY_TIER`. DMem **never auto-detects and never silently
downgrades** — a misconfigured tier fails loudly at startup.

| Tier | Backends | Setup | Retrieval quality |
|---|---|---|---|
| `sqlite` *(default)* | SQLite + brute-force vectors; facts as validity-window rows | none | good for personal / single-project scale |
| `consolidated` | one graph DB with native vectors (Neo4j 5.11+ or FalkorDB) | one server | middle ground, real graph traversal |
| `pro` | dedicated Postgres+pgvector (documents) + Graphiti-backed graph (facts, bi-temporal) | two servers | best |

The **interface is identical** at every tier — only quality and infra differ.
See [`.env.example`](.env.example) for all config.

## Distribution surfaces (all thin adapters over one core)

- **Python SDK** — `from dmem import DMemEngine` (above).
- **MCP server** — `dmem-mcp` (stdio). Exposes `dmem_recall`, `dmem_remember`,
  `dmem_handoff`, `dmem_ingest_document`, `dmem_forget` to any MCP client
  (Claude Desktop, Cursor, …). See [`docs/mcp.md`](docs/mcp.md).
- **OpenAI-compatible proxy** — point your app's base URL at DMem and get memory
  injection + history compaction transparently, no code change:
  `uvicorn dmem.adapters.proxy:app`. See [`docs/proxy.md`](docs/proxy.md).
- **IDE extension** — scaffold in [`extensions/vscode/`](extensions/vscode);
  wraps the MCP server. (Work in progress.)

No memory logic is duplicated per surface.

## What's built in (every tier, degrading gracefully)

- **Hybrid retrieval** — vector + keyword + graph, fused with reciprocal-rank
  fusion.
- **Typed fact extraction** — categorized (preference / decision / project fact
  / identity / credential / …), never freeform blobs.
- **Bi-temporal facts** — a contradicted fact is *closed* (validity window), not
  deleted; full history is queryable.
- **Explicit conflict surfacing** — changed facts are flagged in the handoff,
  not silently collapsed to the newest value.
- **Provenance on every fact** — source, timestamp, origin.
- **Recency-weighted scoring** — a stale fact doesn't outrank a newer one.
- **Cross-encoder reranking** — a lightweight relevance re-order, *not* an LLM
  call (falls back to lexical when the model isn't installed).
- **Structure-aware chunking** — respects document concept/section boundaries;
  tables are kept whole.
- **Token-budgeted handoff** — hard cap; lowest-priority facts drop first,
  never arbitrary truncation. Credentials are withheld from the handoff blob.
- **Cardinality-aware contradictions** — single-valued predicates supersede on
  change; multi-valued ones accumulate (both "uses Postgres" and "uses Redis"
  stay current). Configurable per predicate.
- **Embedding-model guard** — the store records which embedder built it; a
  silent model swap is caught (warn/error) instead of corrupting retrieval.
  `reembed()` rebuilds vectors after an intentional change.
- **Credential policy** — extracted secrets are redacted-and-not-embedded by
  default (`CREDENTIAL_POLICY`); GDPR deletes via `forget_fact` /
  `forget_matching` / `forget`.
- **Ingestion dedup** — re-uploading the same document doesn't bloat the store.
- **Escape hatch** — retrieval is exposed as a tool (`dmem_recall`) so the model
  can pull more context mid-conversation instead of betting on a one-shot
  handoff.
- **Hard fallback** — empty / low-confidence retrieval is flagged so the caller
  triggers the escape hatch or surfaces the gap, never proceeds silently.
- **Fail-open** — retrieval errors degrade to empty (with a warning) rather than
  breaking the host app.

## Ordering decisions (called out because the spec left them open)

- `fuse → recency-weight → rerank → cut`. Recency matters at candidate-selection
  time (so newer contradicting facts win); the cross-encoder then re-orders the
  recency-aware pool by relevance and we blend, not overwrite, the scores.
- Under budget pressure the handoff drops lowest-priority items; if a low/empty
  confidence handoff is requested, the `low_confidence` flag fires so the escape
  hatch takes over.

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/open-questions.md`](docs/open-questions.md).

## Retrieval-quality smoke test

Get a day-one signal on retrieval quality for your config (and measure the lift
from plugging in a real embedding endpoint):

```bash
dmem-eval            # hit@k / recall@k / MRR + behavioral checks; exit 0/1 for CI
dmem-eval --json
```

Lighter than formal benchmarking (which stays deferred) — see
[`docs/evaluation.md`](docs/evaluation.md).

## Status & scope

Alpha. Formal benchmarking (LongMemEval / LoCoMo) is **deferred** until there's
real traffic to tune against, per the source spec — the built-in safeguards
above and the `dmem-eval` smoke harness are present from day one, threshold
calibration is not. Not in scope: billing, hosted proxy, provider markup — DMem
is a self-contained library.

## Development

```bash
pip install -e ".[dev]"
pytest                     # offline suite — no external services needed
```

Graph-tier (Neo4j/FalkorDB) and pro-tier integration tests are gated behind env
vars and skipped by default — see [`docs/testing.md`](docs/testing.md) for how to
run them against a live server.

MIT licensed.
