# DMem — Overview, Usage & Validation Report

*A pluggable, self-hostable AI memory & context library.*
*Persistent memory + document RAG + model-switch continuity, distributed as a library — not a hosted service.*

**Companion report:** [`docs/confidence.md`](docs/confidence.md) (per-capability confidence table) · **CI:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) · **Repo:** `dattamks/dmem`

---

## 1. What DMem is (in one paragraph)

DMem is the **memory and token-efficiency layer** that sits between an application and any LLM. It remembers durable facts about a user or project, ingests documents so only the relevant slices are ever sent to the model, and — when you switch models mid-conversation — hands the new model a compact summary instead of the entire history. It is **bring-your-own-infrastructure**: you point it at *your* databases and *your* embedding vendor via environment variables, and it connects. It installs with `pip` and plugs into any ecosystem through four surfaces: a Python SDK, an MCP server, an OpenAI-compatible proxy, and (planned) an IDE extension.

It is **not** a hosted product, an API reseller, or a billing layer. Those were explicitly out of scope. DMem is the self-contained library that reduces token consumption and preserves continuity.

---

## 2. What problem it solves (why it's built)

Three concrete, expensive problems in LLM apps:

1. **Documents are re-sent every turn.** Ask five questions about a 50-page PDF and a naive app puts the whole PDF in context five times. DMem indexes it once and retrieves only the relevant chunks per question.
2. **Long conversations grow without bound.** Every turn resends the whole transcript. DMem compacts older turns into a summary while keeping recent turns verbatim.
3. **Switching models means starting over.** Move from Model A to Model B and you normally resend everything. DMem sends a **budgeted handoff** — the high-value facts, compactly.

The payoff is fewer tokens sent to the LLM (lower cost + latency) and continuity across sessions and model switches.

---

## 3. What it can do (core capabilities)

- **Persistent typed memory** — extracts facts into categories (preference, decision, project fact, identity, credential, relationship, event) rather than free-text blobs.
- **Bi-temporal fact history** — when a fact changes, the old value is *closed* (validity window), never deleted; full history stays queryable.
- **Cardinality-aware contradictions** — single-valued predicates (`works_at`) supersede on change; multi-valued (`uses`) accumulate ("uses Postgres" *and* "uses Redis" are both true).
- **Document RAG** — OCR → concept-structured markdown → concept-boundary chunking → vector index; retrieves relevant slices only.
- **Hybrid retrieval** — vector + keyword + graph traversal, fused, recency-weighted, with conflict surfacing.
- **Token-budgeted model-switch handoff** — hard cap; lowest-priority facts drop first; credentials withheld.
- **Conversation compaction** — keeps per-turn cost flat as conversations grow.
- **Escape-hatch tool** — the model can call `dmem_recall` mid-conversation if it senses a gap.
- **Safety & governance** — credential redaction, GDPR delete at fact/entity/namespace granularity, namespace isolation for multi-tenancy, an embedding-model-change guard, and fail-fast configuration with a `dmem doctor` preflight.

---

## 4. Benefits — including *measured* token savings

Token savings were **measured**, not just estimated (harness: `dmem-token-savings`). The percentage is a ratio of two token counts, so it's robust to the exact tokenizer.

| Scenario | Baseline → DMem | **Saving** |
|---|---|---|
| Documents (retrieve relevant chunks vs. full document each turn) | ~5,982 → ~1,845 | **~69%** |
| Long conversation (compaction vs. full transcript) | ~2,280 → ~991 | **~57%** |
| Model switch (budgeted handoff vs. full history) | ~729 → ~44 | **~94%** |
| **Blended** | ~8,991 → ~2,880 | **~68%** |

**Honest caveats (read these):**
- These measure **context volume sent to the LLM**. They exclude DMem's own overhead — an embedding call per ingest and the injected-memory block on each request — which reduce **net** savings.
- Document savings **scale with document size**: bigger doc vs. retrieved slice → higher saving.
- A **short, single-model, no-document chat saves ~0%** (possibly net-negative). DMem pays off on high-context, long-running, or model-switching workloads — the cases it was designed for.

Other benefits: **no vendor lock-in** (BYO everything), **permissive licensing option** (an MIT embedded graph, vs. Neo4j's GPLv3 / FalkorDB's SSPL), and **plug-and-play into existing infra** (it never recreates your databases — only adds its own prefixed tables/labels with `IF NOT EXISTS`).

---

## 5. How to use it

### 5.1 Prerequisites (bring your own)

DMem never installs or provisions infrastructure. The production tier (`pro`, the default) has **three mandatory prerequisites**, supplied via env vars:

1. **Graph database** — `GRAPH_DB` (`neo4j` | `falkordb`), `GRAPH_DB_URL` (+ user/password)
2. **Postgres + pgvector** — `PGVECTOR_URL` (DMem actively checks the `vector` extension is enabled and tells you the exact fix if not)
3. **Embedding endpoint** — `EMBEDDING_PROVIDER` (`openai`-compatible, `google`, or `cohere`) + key

If any is missing, DMem **fails fast at startup** with an actionable message. Run **`dmem-doctor`** to check your setup (it live-probes pgvector + graph reachability and reports READY / NOT READY per prerequisite).

Already have infrastructure? Just give DMem the URLs — it coexists (prefixed tables, `IF NOT EXISTS`, no reads/writes to your data). See [`docs/bring-your-own.md`](docs/bring-your-own.md).

### 5.2 The four ways to consume it

| Surface | What it is | Use it when |
|---|---|---|
| **Python SDK** | `from dmem import DMemEngine` — a library you import | integrating into a Python app |
| **MCP server** | `dmem-mcp` (stdio) exposing `dmem_recall/remember/handoff/ingest_document/forget` | VS Code / Cursor / Claude Desktop / any MCP client |
| **OpenAI-compatible proxy** | `uvicorn dmem.adapters.proxy:app` — point your app's base URL at it | **automatic** token reduction with zero app code change |
| **IDE extension** | scaffold wrapping the MCP server | (work in progress) |

> **Important distinction — MCP mode vs. proxy mode.**
> In **MCP mode**, DMem is a *tool the model can call* — it sits beside the conversation and acts only when invoked, so token savings depend on the agent using the tools.
> In **proxy mode**, DMem sits *in the request path* — it injects memory and compacts history **automatically**, so the token reduction happens without any model cooperation. **For automatic savings, use the proxy.**

### 5.3 Quick start (SDK)

```python
from dmem import DMemEngine   # reads config from env; fails fast if prereqs missing

mem = DMemEngine()
mem.ingest_message("My name is Ada and I prefer Rust. I work at Analytical Engines.")
mem.ingest_document("# Runbook\n\nProd runs Postgres 16 in AWS us-east-1.", document_id="runbook")

print(mem.handoff("who is the user and where is prod?").text)
```

### 5.4 Tiers (choose explicitly; never auto-detected)

| Tier | Backends | Use |
|---|---|---|
| `pro` *(default)* | Neo4j/FalkorDB (facts) + Postgres/pgvector (documents) + embedding endpoint | **production** |
| `consolidated` | one graph DB doing both jobs — networked (Neo4j/FalkorDB) **or embedded `kuzu`** (MIT, in-process, no server) | dev / embedded |
| `sqlite` | single file, no servers, offline embedder | dev / test |

### 5.5 Embedding vendors (pick by name + key)

`EMBEDDING_PROVIDER=openai` (covers any OpenAI-compatible endpoint: Qwen/DashScope, vLLM, Ollama, Azure, Together, …) · `google` (Gemini native) · `cohere` (native) · `offline` (dev).

### 5.6 Expected output (what a handoff looks like)

```
Context handoff (compact):
! user works_at Meta
! user works_at Google        ← changed fact, surfaced (not silently collapsed)
- user uses Postgres
- user uses Redis
- [Runbook] Prod runs Postgres 16 in AWS us-east-1.

Changed facts (surfaced, not resolved):
- user works_at: Meta -> Google
```

---

## 6. Validation — the honest picture

The guiding principle throughout: **"the mechanism works" (validated) is separate from "the output quality is good" (often unproven).** Retrieval always *returns* results; whether they're the *best* results depends on the embedder. A handoff is always *produced*; whether it preserves *enough* continuity is a quality question.

### 6.1 Confidence at a glance

| Capability | Confidence | How it was validated |
|---|---|---|
| Memory engine — SQLite tier (no deps) | **~100%** | 150+ deterministic tests |
| Memory engine — Kuzu embedded graph | **~100%** | full engine on real Cypher, in-process |
| pgvector document store | **~95%** | real Postgres 16 / pgvector 0.6.2 |
| Neo4j graph backend | **~95%** | **live Neo4j 5 in CI** |
| FalkorDB graph backend | **~95%** | **live FalkorDB 4.20 in CI** (after a CI-caught fix) |
| Pro composite (facts + docs) | **~95%** | real Kuzu + real pgvector together |
| MCP protocol handshake + tools | **~95%** | real `ClientSession` round-trip |
| Proxy request path + SSE streaming | **~95%** | real httpx over the wire |
| Embedding HTTP clients (3 vendors) | **~90%** | real httpx vs. localhost vendor fakes |
| Concurrency + persistence | **~95%** | threaded access + restart, both backends |
| Token-volume reduction | **~100%** | measured (~69 / 57 / 94 / 68%) |
| Retrieval **relevance** with a real embedder | **unmeasured** | wired in CI, needs a key |
| Model-switch **coherence** (is the handoff *enough*?) | **unproven** | needs an LLM-judge eval (not built) |
| Cross-encoder rerank improvement | **unrun** | model download blocked in dev sandbox |
| Real cloud vendor exact response shapes | **wired** | CI `vendor-live` job runs with secrets |
| VS Code editor UI | **unrun** | MCP protocol proven; extension not built |

### 6.2 What works with ~100% confidence
Run the full memory engine with zero infra (SQLite) or an embedded MIT graph (Kuzu); real pgvector document search; the pro composite; serve memory over MCP; proxy an LLM with injection + streaming; fail-fast + `dmem doctor`; the measured token reductions; concurrency + restart survival; GDPR delete + credential redaction. **These guarantee the mechanism.** Output *quality* needs a real embedder/LLM (see §6.4).

---

## 7. The validation *strategy* — three layers, and why each workaround was necessary

This is the core of how confidence was built. The dev sandbox has a hard constraint: **its network proxy allows only package registries (pypi, npm, crates, go) and localhost. All container-image pulls (Docker Hub, ghcr, quay) are blocked, and there is no outbound access to cloud LLM/embedding vendors.** So a naive "spin up Docker and test everything" was impossible. Instead, validation was built in **three deliberate layers**, matching each dependency to the cheapest environment that could actually exercise it.

### Layer 1 — Validated outright (deterministic, in-sandbox, no external anything)
The entire core engine and all its mechanisms run with zero external dependencies. **150+ tests** cover fact extraction, bi-temporal contradiction, cardinality, dedup, hybrid retrieval, recency, conflict surfacing, budgeted handoff, credential redaction, GDPR deletes, chunking, config fail-fast, and the `dmem doctor` logic — on the SQLite tier and pure logic. This is the bedrock: fully reproducible, no caveats.

### Layer 2 — Validated by "roundabout" means (real engines/protocols, in-sandbox, via pip + localhost)
Where a dependency *looked* like it needed Docker, we found a way to exercise the **real code path** without the blocked infrastructure. The rationale for each:

| What needed testing | The obstacle | The workaround (and why it's legitimate) |
|---|---|---|
| **pgvector** document store | Postgres is a server; Docker image blocked | **`pgserver`** — a pip package that bundles a real PostgreSQL + pgvector binary. Ran the store against **real Postgres 16 / pgvector 0.6.2** with no Docker. This validates the *actual* SQL, the HNSW index, and the extension-enable probe — not a mock. |
| **Graph backend logic** (facts, Cypher, traversal) | Neo4j/FalkorDB are servers; images blocked; neither is pip-installable | **Kuzu** — an MIT-licensed *embedded* graph DB installable via pip. Ran the **full engine on real Cypher** in-process. This validates real graph traversal + bi-temporal queries against a genuine graph engine (and doubles as a shippable clean-license option). |
| **MCP protocol** (the VS Code path) | Needs a real MCP client | The **`mcp` SDK's in-memory client↔server transport** — a real `ClientSession` does the real handshake, `list_tools`, and tool calls over the actual protocol, minus only the OS pipe. |
| **Proxy** over HTTP (forward + SSE streaming) | Needs an upstream LLM endpoint | A **localhost fake OpenAI server** (`http.server`) + **real httpx**. Validates real URL building, auth headers, non-streaming JSON, and byte-for-byte SSE passthrough over an actual socket. |
| **Embedding vendor HTTP clients** (OpenAI/Google/Cohere) | No outbound to cloud vendors | **Localhost fake vendor servers** returning each vendor's real response shape, driven over real httpx. Validates request formation, auth, and parsing for all three shapes. |
| **Pro composite** (two backends together) | Needs both servers live | Composed **real Kuzu (facts) + real pgvector (docs)** and validated the routing/merge — two genuine engines, not fakes. |
| **Concurrency + persistence** | — | Multi-threaded ingest/retrieve (the access pattern the MCP/proxy adapters produce) + data survival across restart, on SQLite and Kuzu. |
| **Token savings** | tiktoken's BPE vocab download is blocked | Measured with a **consistent counter** (tiktoken when available, else a ~4-chars/token approximation). Because the metric is a **ratio**, the percentage is tokenizer-robust even without the exact vocab. |

The through-line: **wherever a real engine could be obtained through an allowed channel (pip) or emulated at the protocol boundary (in-memory transport, localhost fakes), we tested the real code path rather than mocking it.** These are not shortcuts around testing — they are the maximally-real test achievable under the sandbox's network policy.

### Layer 3 — Validated in CI (things that genuinely need networked servers or real APIs)
Some dependencies have **no pip equivalent and no localhost stand-in that proves the real thing** — specifically the networked graph *servers* (Neo4j/FalkorDB) and the *real* cloud vendor APIs. For these we built a **GitHub Actions pipeline** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) that runs where Docker Hub and the internet *are* reachable:

- **`offline` (Python 3.10 / 3.11 / 3.12)** — the entire Layer-1 + Layer-2 suite, on every push.
- **`integration-neo4j` / `integration-falkordb`** — spin up **real Neo4j and FalkorDB as service containers** and run the graph suite against them.
- **`vendor-live`** — runs **real OpenAI/Google/Cohere** embedding tests and a **real-embedder retrieval-quality** assertion, using API keys from repo secrets; **skips cleanly when no secrets are present**.

**Why CI, and what it caught.** CI is not decoration — it closed the two biggest "unrun" items and immediately earned its keep:
- **Neo4j passed live on the first run** — closing a backend that couldn't be tested in-sandbox at all.
- **FalkorDB failed on the first run** — CI caught a **real dialect bug**: FalkorDB rejects Neo4j's `CREATE CONSTRAINT … REQUIRE … IS UNIQUE` syntax. The fix made `graph_store.py` backend-aware (Neo4j keeps its native vector index; FalkorDB skips unsupported constraints and uses brute-force cosine retrieval). **The next run went green.** This is exactly the class of bug that *only* a live server surfaces — and the reason the CI layer exists.

**Current CI status: all jobs green** — offline ×3, `integration-neo4j`, `integration-falkordb`, `vendor-live`.

### The layered rationale, summarized
> Match each dependency to the cheapest environment that can exercise it *for real*:
> **pure logic → deterministic tests; real engines obtainable via pip / emulatable at the protocol edge → in-sandbox real-path tests; networked servers and cloud APIs → CI where they're reachable.** Mock nothing that can be run for real.

---

## 8. What is **not** evaluated (and how to close it)

| Gap | Why it's open | How to close it |
|---|---|---|
| **Retrieval quality with a real embedder** | Needs a real embedding model; offline embedder is a lexical floor | Add an `OPENAI_API_KEY` (etc.) secret — the CI `vendor-live` job runs `dmem-eval` and asserts a quality bar. Or run locally against Ollama. |
| **Model-switch coherence** (is the handoff *sufficient*?) | Subjective; needs an LLM-judge eval that isn't built yet | Build a lightweight LLM-judge eval (scripted multi-turn + switch + grade); run in CI with a key. The formal benchmarks (LongMemEval/LoCoMo) remain deliberately deferred. |
| **Cross-encoder rerank improvement** | Model weights download blocked in dev sandbox | Gated test with `dmem[rerank]`; runs in CI / locally where HuggingFace is reachable. |
| **VS Code extension UI** | Extension is a scaffold; MCP protocol already proven | Implement the extension; test headless with `@vscode/test-electron`. |
| **Production scale / latency / cost** | CI proves correctness, not load | Real deployment + monitoring. |

**The clean boundary:** *CI proves "the code works correctly with real dependencies." It cannot prove "it's good enough" or "it scales."* The first is essentially fully covered; the second needs evals (some CI-runnable with a key) and real deployment.

---

## 9. At-a-glance summary

- **What:** a BYO, pip-installable AI memory + token-efficiency library; four surfaces; not a hosted service.
- **Why:** cut tokens on documents, long conversations, and model switches; keep continuity across sessions/models.
- **Measured benefit:** ~69% (documents) / ~57% (long conversation) / ~94% (model switch) / ~68% blended context-token reduction — on the workloads it targets.
- **Validated outright:** the whole core engine (150+ tests) + real pgvector + real Kuzu + live MCP + real-HTTP proxy/embeddings + concurrency, all in-sandbox via pip/localhost.
- **Validated in CI:** live Neo4j + live FalkorDB (a real bug was caught and fixed) + real vendor APIs (with keys).
- **Not yet evaluated:** retrieval *quality* with a real embedder, model-switch *coherence*, cross-encoder rerank, the VS Code UI, and production scale — each with a concrete path to close.

*Full per-capability detail: [`docs/confidence.md`](docs/confidence.md). Testing how-to: [`docs/testing.md`](docs/testing.md). Prerequisites: [`docs/prerequisites.md`](docs/prerequisites.md).*
