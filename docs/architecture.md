# DMem Architecture

DMem is one core engine (`dmem.engine.DMemEngine`) wrapped by thin adapters. No
memory logic lives in an adapter.

```
                 ┌───────────────── distribution surfaces ─────────────────┐
                 │  Python SDK      MCP server        IDE extension         │
                 │  (import)        (dmem-mcp)        (wraps MCP)           │
                 └───────────────────────┬─────────────────────────────────┘
                                         │  (thin adapters only)
                              ┌──────────▼───────────┐
                              │     DMemEngine       │
                              └──────────┬───────────┘
        ingest_message / ingest_document │ retrieve / handoff / compact / forget
             ┌───────────────────────────┼───────────────────────────┐
             ▼                           ▼                           ▼
        Pipeline                    Retrieval                    Providers
   OCR→OKF→chunk→extract→dedup   hybrid fuse→recency→rerank    embeddings / rerank
                                   →conflict→handoff            / OCR / LLM (BYO)
             └───────────────┬───────────┴───────────────┬───────────┘
                             ▼                           ▼
                        MemoryStore (one interface, three tiers)
             sqlite  │  consolidated (graph+vectors)  │  pro (pgvector + graph)
```

## Ingestion

**Messages** → typed fact extraction (heuristic offline, or BYO LLM) → each fact
gets provenance → contradiction handling → dedup → embed → store.

**Documents** → (OCR if binary, via `OCR_HOST_URL`) → OKF structuring (split by
heading into concept sections with metadata) → structure-aware chunking (respect
concept boundaries, keep tables whole) → dedup by content hash → embed → store.

### Contradiction handling (bi-temporal + cardinality)

On `ingest_message`, for each candidate fact:

1. **Exact dedup** — identical `(namespace, subject, predicate, object)` already
   current → skip.
2. **Cardinality check** — resolve the predicate's cardinality (registry +
   `MULTI_VALUED_PREDICATES`/`SINGLE_VALUED_PREDICATES` overrides + the
   extractor's optional `replaces` flag):
   - **SINGLE-valued** (`works_at`, `has_name`, …) or explicit `replaces=True` →
     close prior current values (`valid_to = now`) and store the new one with
     `supersedes` linking the old. Nothing is deleted; history stays queryable.
   - **MULTI-valued** (`uses`, `prefers`, …) → **accumulate**; both values stay
     current. Unknown predicates default to SINGLE.

Conflict surfacing (at retrieval) fires only when a value was genuinely
superseded — i.e. a closed validity window exists — so concurrent multi-valued
facts are not mislabeled as conflicts.

### Deletion (GDPR)

- `forget(namespace)` — wipe a whole tenant.
- `forget_fact(fact_id)` — hard-delete one fact, including its closed history.
- `forget_matching(namespace, subject=…/predicate=…/object=…)` — erase every
  fact about an entity. These hard-delete and override the preserve-history
  default (distinct from the bi-temporal *close*).

## Retrieval

`fuse → recency-weight → conflict-flag → rerank → cut` (see
[open-questions.md](open-questions.md) for why this order).

- **Three channels**: vector (cosine over embeddings), keyword (token overlap /
  full-text), graph (entity-neighborhood). Independent; fused via weighted
  reciprocal-rank fusion (robust to different score scales).
- **Recency** multiplies fused scores by a half-life decay (blended, floor 0.5,
  so it nudges rather than dominates).
- **Reranking** re-orders the top candidates with a cross-encoder (or lexical
  fallback); the rerank score is blended `0.7*rerank + 0.3*prior`.

## Handoff

Token-budgeted, priority-ordered. Priority = fact-type importance + confidence +
retrieval score. Over budget → lowest-priority items drop first (recorded in
`Handoff.dropped`). Conflicts are surfaced in a dedicated section. Credentials
are withheld from the text. Empty/low-confidence → `low_confidence=True` for the
hard-fallback path.

## Tiers & the store interface

Every backend implements `dmem.stores.base.MemoryStore`: facts (upsert / dedup
lookup / current-facts / close / get), documents (upsert / exists), and three
retrieval channels (`vector_search`, `keyword_search`, `graph_neighbors`), plus
`delete_namespace` for GDPR. The SQLite tier approximates graph traversal with a
one-hop co-occurrence lookup so the fusion code path is identical everywhere.

## Multi-tenancy

Every stored item carries a `namespace`. Pass `namespace=` to isolate users /
tenants. `forget(namespace)` hard-deletes everything for one tenant.
