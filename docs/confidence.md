# Confidence & Validation Report

What has actually been *run* vs. estimated. The distinction that matters
throughout: **"the mechanism works" (validated) is separate from "the output
quality is good" (often unproven).** Retrieval always *returns* results; whether
they're the *best* results depends on the embedder. A handoff is always
*produced*; whether it preserves enough continuity is a quality question.

## Confidence metrics (by capability)

| Capability | Confidence | Basis |
|---|---|---|
| Memory engine on **SQLite tier** (offline, no deps) | **~100%** | 130+ deterministic tests, zero external deps |
| Memory engine on **Kuzu** embedded graph | **~100%** | full engine on real Cypher, pip-only |
| **pgvector** document store ops | **~95%** | real Postgres 16 / pgvector 0.6.2 |
| **Pro composite** (facts + docs together) | **~95%** | ProStore over real Kuzu + real pgvector |
| Contradiction / cardinality / dedup / recency / conflict | **~100%** | deterministic logic, fully tested |
| Token-budgeted handoff, credential redaction, GDPR delete | **~100%** | deterministic, tested |
| **Config fail-fast + pgvector-enabled check + `dmem doctor`** | **~100%** | tested, incl. real extension probe |
| **MCP** protocol handshake + tool calls | **~95%** | real `ClientSession` round-trip |
| **Proxy** request path + SSE streaming | **~95%** | real httpx over the wire |
| Embedding **HTTP client** (OpenAI/Google/Cohere shapes) | **~90%** | real httpx vs. localhost fakes |
| Concurrency (locked stores) + persistence | **~95%** | threaded + restart, both backends |
| **Token-volume reduction** (the mechanism) | **~100%** | measured ~69/57/94/68% |
| — | — | — |
| Retrieval **relevance** with a real embedder | **unmeasured** | only the offline lexical floor run |
| **LLM** fact-extraction quality | **unmeasured** | heuristic works; LLM path model-dependent |
| **Model-switch coherence** (is the handoff *enough*?) | **unproven** | mechanism 100%, sufficiency not evaluated |
| Cross-encoder **rerank** improvement | **unrun** | model download blocked in dev sandbox |
| **Neo4j / FalkorDB** backends | **unrun** | no server available in dev sandbox |
| Real **cloud** vendor API exact shapes | **unrun** | only localhost fakes |
| **VS Code** editor UI integration | **unrun** | MCP protocol proven; extension not built |

## What works with ~100% confidence (validated against real code paths)

These are the things a user can do that **will work**, because the exact path has
been run with passing tests and has no runtime dependency that can fail:

1. **Run the full memory engine with zero infrastructure** (SQLite tier): ingest
   messages, extract typed facts, bi-temporal contradiction handling,
   multi-valued accumulation, dedup, hybrid retrieval, recency-weighted scoring,
   conflict surfacing, token-budgeted handoff, escape-hatch recall.
2. **Run the full engine on an embedded MIT graph** (Kuzu) with real Cypher
   traversal — no server, `pip install` only.
3. **Store & search documents in a real pgvector** (extension auto-checked; clear
   error if missing).
4. **Compose the pro tier** (graph facts + pgvector documents) with merged search.
5. **Serve memory over MCP** — handshake, list tools, recall/remember/handoff/
   ingest/forget over the real protocol.
6. **Proxy an OpenAI-compatible endpoint** — non-streaming and SSE streaming,
   with memory injection and history compaction over real HTTP.
7. **Fail fast & self-diagnose** — mandatory-prerequisite validation and
   `dmem doctor` (including the live pgvector-enabled probe).
8. **Reduce context volume** — the measured ~69% (documents) / ~57% (long
   conversation) / ~94% (model switch) reductions are produced deterministically
   by the mechanisms.
9. **Handle concurrent access and survive restarts** on SQLite and Kuzu.
10. **Delete/forget** at fact, entity, and namespace granularity (GDPR), and
    withhold/redact credentials.

Honest boundary: these guarantee the **mechanism**. Output **quality** (are the
retrieved chunks the *right* ones? is the handoff *sufficient*?) needs a real
embedder/LLM and the evaluations below.

## How to test the currently-unvalidated (network/infra) pieces

They're blocked in *this* sandbox, not in general. Concrete ways to test each —
the CI pipeline (`.github/workflows/ci.yml`) automates most of them:

### 1. Neo4j / FalkorDB (live graph servers)
- **CI service containers (recommended, automated):** the `integration-graph`
  job spins up Neo4j and FalkorDB as GitHub Actions services and runs the
  already-written gated suite (`tests/integration/test_graph_tier.py`). Runs on
  every push — no local setup.
- **Local Docker:** `docker compose up -d falkordb` then
  `DMEM_TEST_GRAPH_KIND=falkordb DMEM_TEST_GRAPH_URL=redis://localhost:6379 pytest tests/integration`.
- **Free hosted:** point `DMEM_TEST_GRAPH_URL` at a Neo4j Aura / FalkorDB Cloud
  free instance — no Docker at all, just a URL.
- **testcontainers:** ephemeral containers spun up from the test itself.

### 2. Real cloud embedding / LLM vendors
- **CI with secrets (automated):** the `vendor-live` job runs
  `tests/integration/test_vendor_live.py` against real OpenAI/Google/Cohere using
  API keys stored as GitHub secrets; it **skips** when secrets are absent, so it's
  safe. This confirms the one thing the localhost fakes can't: the vendor's real
  response shape.
- **Local:** set `OPENAI_API_KEY` (etc.) and run the same tests — they hit the
  real API.
- **Cassettes (`vcrpy`):** record one real response with a key, replay in CI with
  no network — deterministic and free thereafter.

### 3. Retrieval quality with a real embedder (the biggest quality unknown)
- Run `dmem-eval` with a real endpoint: `EMBEDDING_HOST_URL=... dmem-eval`. The
  `vendor-live` CI job does this and asserts `hit@k` clears a bar — turning the
  offline floor into a real quality number.
- Local option with no cloud key: point `EMBEDDING_HOST_URL` at a local **Ollama**
  (`/v1/embeddings`) — real semantic embeddings, no vendor account.

### 4. Model-switch coherence
- Needs an **LLM-judge** eval: script a multi-turn conversation, switch models
  through the handoff, and have an LLM grade whether continuity held. This is the
  lightweight cousin of LongMemEval/LoCoMo (formal benchmarking, still deferred).
  Runs in CI with a key.

### 5. Cross-encoder rerank
- Gated test with `pip install "dmem[rerank]"` + `RERANK_MODEL=...`; runs in CI or
  locally where HuggingFace is reachable (model download).

### 6. VS Code extension UI
- The MCP **protocol** is already proven. The remaining work is the extension
  package itself; test it headless with `@vscode/test-electron` in CI, or manually
  by adding the `dmem-mcp` server to a VS Code MCP config.
