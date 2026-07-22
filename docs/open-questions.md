# Open Questions, Gaps & Edge Cases

Tracked deliberately. Some are resolved by decisions already baked into the code
(marked ✅ with the choice made); the rest are genuinely open (marked ❓) and
need a call from product/eng, ideally informed by real usage.

## Resolved in this build (decisions to review)

- ✅ **Scope of the "plugin".** The memory/context layer (spec §8) plus the thin
  OpenAI-compatible proxy adapter (`dmem.adapters.proxy`) are built. Auth/billing
  remain out of scope.
- ✅ **Transparent proxy adapter.** `dmem.adapters.proxy` sits in front of any
  OpenAI-compatible `/v1/chat/completions`: injects relevant memory / a
  model-switch handoff, compacts long histories before forwarding, and persists
  facts from each exchange — host app only changes its base URL. Pure sync core
  (injectable forwarder) + a framework-free ASGI app.
- ✅ **Proxy streaming.** `stream: true` is supported: request-side memory
  injection + compaction, byte-for-byte SSE passthrough to the client, and the
  assembled assistant reply teed into memory when the stream ends (driven from a
  worker thread so the ASGI loop isn't blocked). Only assistant `content` deltas
  are reconstructed into memory — tool-call/function-arg deltas pass through but
  aren't teed in.
- ✅ **Retrieval ordering** (spec called this open): `fuse → recency → rerank →
  cut`. Recency applies at candidate selection; rerank re-orders within the
  recency-aware pool; scores are blended, not overwritten.
- ✅ **Escape hatch under budget pressure.** The handoff exposes a
  `low_confidence` flag and a `dmem_recall` tool; the model recovers via the
  tool rather than the budgeted handoff trying to be exhaustive. No retry loop
  in the handoff itself (avoids budget thrash).
- ✅ **Failure posture.** `retrieve`/`handoff` **fail open** — a store error
  degrades to empty-with-warning rather than breaking the host app. Reasonable
  default for an embedded library; make it configurable if some adopter wants
  fail-closed.
- ✅ **Zero-setup default.** No `EMBEDDING_HOST_URL` → offline hashing embedder
  with a warning, so `pip install dmem` runs end-to-end. Quality is lexical-only
  until a real endpoint is configured.
- ✅ **Credentials in handoffs.** `credential`-typed facts are withheld from the
  handoff text by default; their existence is noted.
- ✅ **Contradiction granularity (multi-valued predicates).** Facts now carry a
  per-predicate **cardinality**. SINGLE-valued predicates (`works_at`,
  `has_name`, …) supersede on change; MULTI-valued (`uses`, `prefers`, …)
  accumulate — "I use Postgres" and "I use Redis" are both current. Unknown
  predicates default to SINGLE; extend via `MULTI_VALUED_PREDICATES` /
  `SINGLE_VALUED_PREDICATES`, or the LLM extractor's per-fact `replaces` flag.
  Conflict surfacing now fires only on a real supersede (a closed validity
  window exists), so concurrent multi-valued facts aren't mislabeled a conflict.
- ✅ **Single-fact hard delete (GDPR).** `forget_fact(id)` removes one fact
  (including its closed history); `forget_matching(ns, subject=…/predicate=…/
  object=…)` erases everything about an entity. These hard-delete and win over
  the preserve-history default; `forget(ns)` still wipes a whole namespace.
- ✅ **Embedding-model-change guard.** The store stamps the embedding signature
  (model + dim); on open, a mismatch is caught. `EMBEDDING_MISMATCH_POLICY` =
  warn (default) / error / ignore. `engine.reembed()` rebuilds all vectors with
  the current model and updates the signature. A `schema_version` is also
  stamped for future migrations.
- ✅ **Credential storage policy.** `CREDENTIAL_POLICY` = redact (default: store
  the fact, mask the value, never embed it) / drop (don't store) / store (keep
  raw, still withheld from handoffs). Default no longer retains secret values.

## Open — need a decision

### Data / privacy / legal
- ❓ **Credential encryption at rest.** Default is now redact-and-don't-embed
  (resolved above), but the `store` policy keeps raw values in plaintext and
  there's no encryption-at-rest option. Add one before anyone uses `store` in
  production. Also: credential *detection* is a coarse keyword match — it will
  miss secrets that don't say "key/token/password".
- ❓ **PII handling / redaction policy.** No PII detection on ingest. Should the
  extractor redact or flag emails/phone/SSNs? (Same detection-quality caveat as
  credentials.)
- ❓ **Encryption at rest** for the SQLite file and embeddings — not implemented.

### Retrieval quality (all deferred to post-benchmark per spec)
- ✅ **Day-one retrieval signal.** `dmem-eval` (the `dmem.eval` smoke harness)
  reports hit@k / recall@k / MRR plus behavioral checks against a small labelled
  dataset, so adopters can gauge retrieval quality for their config and measure
  the lift from a real embedding endpoint — without the full LongMemEval setup.
  Formal benchmarking (LongMemEval / LoCoMo / ConvoMem) remains deferred.
- ❓ **Confidence threshold default** (`LOW_CONFIDENCE_THRESHOLD=0.15`) is a
  guess. Needs calibration against real retrieval score distributions — note the
  RRF-based fused scores are small in absolute terms, so this threshold is scale-
  sensitive and should be tuned per embedding model. (`dmem-eval` now gives a way
  to observe those distributions.)
- ❓ **Channel fusion weights** (`vector 1.0 / keyword 0.6 / graph 0.8`) and the
  **recency half-life** (30d) are untuned defaults.
- ❓ **Heuristic extractor recall.** The offline pattern-based extractor catches
  common first-person statements only. It will miss most third-person and
  implicit facts — a real LLM extractor (`LLM_HOST_URL`) is strongly recommended
  for production. Is heuristic-only ever acceptable, or should we warn harder?
- ❓ **Mutually-exclusive multi-valued values.** Cardinality fixed the additive
  case, but a MULTI predicate can still hold mutually-exclusive values on one
  dimension (e.g. `prefers dark mode` vs `prefers light mode`). Resolving that
  needs value-level semantics (the LLM `replaces` signal, or dimension tagging);
  today both accumulate and aren't flagged as a conflict. Acceptable for now.

### Graph tier (consolidated / pro)
- ✅ **Driver result-shape bug fixed.** Neo4j returns Record objects (nodes are
  Mappings); FalkorDB returns positional `result_set` rows and its Node has no
  `__getitem__` (only `.properties`). `_CypherDriver.run()` now normalizes both
  to uniform dict rows; covered offline with fake drivers. All read paths in
  `graph_store.py` were rewritten against the normalized shape.
- ❓ **Vector-index dialect not validated live.** Neo4j (`db.index.vector.*`,
  `CREATE VECTOR INDEX … OPTIONS`) and FalkorDB (`db.idx.vector.*`, its own DDL)
  differ. Both branches are implemented in `_CypherDriver.create_vector_index` /
  `vector_query` from documented syntax but **need validation against a live
  server** — that's what `tests/integration` is for (gated on `DMEM_TEST_GRAPH_URL`).
- ✅ **pgvector validated for real (no Docker).** `tests/integration/
  test_pgvector_tier.py` uses `pgserver` (pip-bundled Postgres+pgvector) to run
  the document store end to end — extension enable, HNSW vector search, full-text
  keyword search, dedup, namespace isolation/delete, meta, table-prefix
  isolation. The `ensure_pgvector` probe and PgVectorStore are no longer stub-
  only. (Confirmed against Postgres 16 / pgvector 0.6.2.)
- ❓ **Graph half + combined pro engine.** The graph store (facts) still needs a
  real Neo4j/FalkorDB (image pulls are blocked in this sandbox; run the gated
  suite where reachable). A single combined pro-tier engine test (graph +
  pgvector together) is still to add.

### Product stance: BYO-only, pro is the product
- ✅ **Mandatory prerequisites, fail fast.** `pro` is the default and the only
  supported production tier: a graph DB + Postgres/pgvector + an embedding
  endpoint are all mandatory BYO env vars. Missing any → `ConfigError` at startup
  with an actionable message. DMem never installs/provisions/bootstraps — the
  bootstrap idea was dropped by decision.
- ✅ **Active pgvector check.** On connect, DMem verifies the `vector` extension
  is enabled, tries to enable it, and otherwise fails with the exact fix
  (`CREATE EXTENSION vector;` / install pgvector). `ensure_pgvector` is unit-
  tested via a fake connection.
- ✅ **`dmem-doctor` preflight.** Inspects env, reports READY/NOT-READY per
  prerequisite with fixes, live-probes pgvector + graph reachability, redacts
  credentials, exits 0/1. MCP server prints it to stderr instead of a traceback
  when prereqs are missing.
- ✅ **`sqlite`/`consolidated` demoted to dev/test only** (explicit opt-in,
  flagged by doctor). Retained for the offline test suite and local dev.

### Bring-your-own infrastructure (plug-and-play)
- ✅ **Existing databases.** DMem points at an existing Postgres+pgvector / graph
  DB and creates only its own prefixed objects with `IF NOT EXISTS` — never
  touching adopter tables. `PGVECTOR_TABLE_PREFIX` makes the pgvector prefix
  configurable (avoid collision / multi-instance). Documented in
  `docs/bring-your-own.md`.
- ✅ **Multi-vendor embeddings.** `EMBEDDING_PROVIDER` selects the API shape:
  `openai` (OpenAI-compatible — Qwen/DashScope, vLLM, TEI, Ollama, Azure, …),
  `google` (Gemini native), `cohere` (v2 native), `offline`. Vendor + key, no
  local model.
- ❓ **More native embedding adapters.** Voyage, Jina, Vertex-native, Bedrock
  embeddings aren't native adapters yet (most work via the OpenAI-compatible
  path if the vendor offers one). Add on demand.
- ❓ **Graph label namespacing for shared Neo4j.** FalkorDB isolates via a named
  graph; a shared Neo4j database relies on a dedicated `GRAPH_DB_DATABASE` since
  labels (`Fact`/`Entity`/`Chunk`) aren't prefixed. A label-prefix option would
  let DMem share one Neo4j database without a dedicated DB.
- ❓ **Cohere asymmetric input types.** Uses `search_document` for everything;
  `search_query` for queries would improve retrieval — needs query-vs-store
  awareness in the embed path.

### Packaging / deployment
- ✅ **Docker packaging.** `Dockerfile` (MCP server / proxy / eval entrypoints,
  non-root, `/data` volume) + `docker-compose.yml` (FalkorDB + proxy by default;
  Neo4j + pgvector under the `pro` profile) + `docs/docker.md` covering SDK vs
  MCP vs proxy, MCP-over-`docker run -i`, and running the graph integration
  suite against the Compose stack. Compose config validated.

### Scale / operational
- ❓ **SQLite vector search is O(n)** brute force. Fine for personal scale; at
  ~10⁵+ items it needs `sqlite-vec`/`vec0` ANN or a push to a higher tier. When
  do we wire in a real ANN index for the SQLite tier?
- ❓ **Concurrency / cross-store consistency.** SQLite tier is single-file with a
  lock. Pro tier writes facts (graph) and chunks (pgvector) in separate
  transactions — a partial failure can leave them inconsistent. No 2-phase /
  outbox yet.
- ❓ **Schema migration / versioning.** A `schema_version` is now stamped in the
  store, but there is no migration *runner* yet — nothing consumes the version to
  transform an old store to a new layout. Still needs an export/import format and
  a migration path before 1.0.

### Product / packaging
- ❓ **Public name.** Repo is `dmem`; confirm that's the shipping name.
- ❓ **License audit of dependencies.** Neo4j Community is **GPLv3** — matters for
  adopters who embed the consolidated/pro tier; FalkorDB, sentence-transformers,
  pgvector licenses should be documented in a `NOTICE`. DMem core itself is MIT
  and pulls none of these unless the adopter opts into an extra.
- ❓ **OCR/OKF as a separate module.** Document ingestion pulls in a GPU OCR
  model for binary docs. Should it be a separate optional package
  (`dmem-documents`) so the conversational-memory core stays tiny?
- ❓ **IDE extension depth.** `extensions/vscode` is a scaffold that wraps the
  MCP server. How much native UI (memory browser, conflict viewer) is in scope?
- ❓ **Async API.** The core is synchronous. The MCP adapter wraps it in async;
  high-throughput hosts may want a native async engine.

## Not doing (explicitly out of scope)
- Billing, metering, provider markup, hosted proxy — DMem is a library.
- Formal benchmarking (LongMemEval / LoCoMo / ConvoMem) — deferred until there
  is a working system with real traffic to tune against.
