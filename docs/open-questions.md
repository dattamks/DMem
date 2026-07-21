# Open Questions, Gaps & Edge Cases

Tracked deliberately. Some are resolved by decisions already baked into the code
(marked ✅ with the choice made); the rest are genuinely open (marked ❓) and
need a call from product/eng, ideally informed by real usage.

## Resolved in this build (decisions to review)

- ✅ **Scope of the "plugin".** Only the memory/context layer (spec §8) is built.
  Proxy/auth/billing are out of scope. A thin OpenAI-compatible proxy adapter
  can be added later without touching the core.
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

## Open — need a decision

### Data / privacy / legal
- ❓ **Credential fact type at all.** We *tag and withhold* credentials, but
  should DMem store them at all? Options: (a) store tagged + encrypted at rest,
  (b) redact-and-drop, (c) store but never embed. Currently stored tagged and
  embedded — **revisit before any production use.** No encryption-at-rest yet.
- ❓ **GDPR vs. bi-temporal "never delete".** Contradicted facts are *closed*,
  not deleted — good for history, but a hard right-to-be-forgotten request must
  win. `forget(namespace)` hard-deletes a whole tenant; there is **no
  single-fact hard delete** yet. Add one?
- ❓ **PII handling / redaction policy.** No PII detection on ingest. Should the
  extractor redact or flag emails/phone/SSNs?
- ❓ **Encryption at rest** for the SQLite file and embeddings — not implemented.

### Retrieval quality (all deferred to post-benchmark per spec)
- ❓ **Confidence threshold default** (`LOW_CONFIDENCE_THRESHOLD=0.15`) is a
  guess. Needs calibration against real retrieval score distributions — note the
  RRF-based fused scores are small in absolute terms, so this threshold is scale-
  sensitive and should be tuned per embedding model.
- ❓ **Channel fusion weights** (`vector 1.0 / keyword 0.6 / graph 0.8`) and the
  **recency half-life** (30d) are untuned defaults.
- ❓ **Heuristic extractor recall.** The offline pattern-based extractor catches
  common first-person statements only. It will miss most third-person and
  implicit facts — a real LLM extractor (`LLM_HOST_URL`) is strongly recommended
  for production. Is heuristic-only ever acceptable, or should we warn harder?
- ❓ **Contradiction granularity.** Same `subject+predicate`/different `object`
  is treated as a contradiction. This is too coarse for multi-valued predicates
  (e.g. "I use Postgres" *and* "I use Redis" are both true, not a contradiction).
  Need a notion of single- vs multi-valued predicates.

### Scale / operational
- ❓ **SQLite vector search is O(n)** brute force. Fine for personal scale; at
  ~10⁵+ items it needs `sqlite-vec`/`vec0` ANN or a push to a higher tier. When
  do we wire in a real ANN index for the SQLite tier?
- ❓ **Concurrency / cross-store consistency.** SQLite tier is single-file with a
  lock. Pro tier writes facts (graph) and chunks (pgvector) in separate
  transactions — a partial failure can leave them inconsistent. No 2-phase /
  outbox yet.
- ❓ **Schema migration / versioning.** No migration story for the SQLite schema
  or the graph model across DMem upgrades. OSS stores live for years — needs an
  export/import format and a migration runner before 1.0.
- ❓ **Embedding-model change.** Vectors from one embedding model are
  incompatible with another. Switching `EMBEDDING_MODEL` silently degrades
  retrieval. Need to store the model id per vector and warn / re-embed on change.

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
