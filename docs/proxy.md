# DMem as an OpenAI-compatible proxy

The proxy adapter adds memory **transparently**: your app keeps using the OpenAI
SDK and only changes its base URL. No code change, no new client library.

```
your app ──► DMem proxy ──► your LLM provider
             │  injects relevant memory / a model-switch handoff
             │  compacts long history before forwarding
             └► persists facts from each exchange
```

## Run it

```bash
pip install "dmem[http]"                 # httpx forwarder; add [mcp] etc. as needed
export PROXY_UPSTREAM_URL=https://api.openai.com/v1
export PROXY_UPSTREAM_API_KEY=sk-...
export EMBEDDING_HOST_URL=...            # for real retrieval quality (optional)
uvicorn dmem.adapters.proxy:app --port 8000   # any ASGI server works
```

Point your client at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "remind me where I work"}],
    user="alice",                         # becomes the memory namespace
)
```

Namespace resolution: the `X-DMem-Namespace` header, else the request's `user`
field, else the engine default. Use it to isolate tenants/users.

## What it does per request

1. **Model-switch detection** — compares `model` to the last one seen for the
   namespace. On a switch, injects a compact **handoff**; otherwise injects
   relevant remembered facts (if any) as a system message. Credentials are never
   injected.
2. **History compaction** — when the forwarded message list exceeds
   `PROXY_COMPACT_OVER_TOKENS`, older turns are replaced with a summary before
   hitting the upstream, cutting tokens actually sent. The last
   `PROXY_KEEP_LAST_TURNS` turns are kept verbatim.
3. **Forward** to your upstream OpenAI-compatible endpoint.
4. **Record** — the newest user turn and the assistant reply are ingested into
   memory so they persist across sessions and model switches. Memory-write
   failures never break the proxied call.

## Configuration (env)

| Var | Default | Meaning |
|---|---|---|
| `PROXY_UPSTREAM_URL` | — | your OpenAI-compatible endpoint (or `OPENAI_BASE_URL`) |
| `PROXY_UPSTREAM_API_KEY` | — | upstream key (or `OPENAI_API_KEY`) |
| `PROXY_INJECT_MEMORY` | `true` | inject memory / handoff |
| `PROXY_COMPACT_HISTORY` | `true` | compact long histories before forwarding |
| `PROXY_COMPACT_OVER_TOKENS` | `3000` | compaction threshold |
| `PROXY_KEEP_LAST_TURNS` | `6` | turns kept verbatim after compaction |
| `PROXY_MEMORY_TOKEN_BUDGET` | `800` | budget for injected handoff / summary |

Plus all the standard DMem env (tier, embeddings, etc.) — see
[`../.env.example`](../.env.example).

## Programmatic use

The orchestration core is a plain, synchronous object with an injectable
forwarder — handy for embedding in your own server or for tests:

```python
from dmem import DMemEngine
from dmem.adapters.proxy import MemoryChatProxy, ProxyConfig

proxy = MemoryChatProxy(DMemEngine(), ProxyConfig.from_env())
response = proxy.complete(openai_request_dict, namespace="alice")
```

## Streaming

Streaming (`stream: true`) is fully supported: memory is injected on the request
side and history compaction still applies, the upstream SSE stream is passed
through to the client **byte-for-byte** (so it looks like a normal OpenAI
stream), and the assembled assistant reply is **teed into memory** once the
stream completes. The response is `text/event-stream`. The blocking generator is
driven from a worker thread so it doesn't stall the ASGI event loop.

## Limitations (v1)

- Compaction summarization uses your configured LLM if set, else an extractive
  fallback (lower fidelity).
- The proxy is synchronous per request (run behind an ASGI server that gives you
  concurrency); heavy deployments may want a native-async engine (tracked in
  open questions).
- Non-`content` streamed deltas (tool-call fragments, function args) are passed
  through to the client but not reconstructed into memory — only assistant
  `content` text is teed in.
