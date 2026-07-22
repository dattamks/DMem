"""Streaming path for the OpenAI-compatible proxy (SSE passthrough + tee)."""

import asyncio
import json

from dmem.adapters.proxy import (MemoryChatProxy, ProxyConfig, create_app,
                                 _SSEAccumulator)


def _sse_chunks(texts, model="gpt"):
    """Build OpenAI-style SSE byte chunks for a sequence of delta contents."""
    chunks = []
    for t in texts:
        obj = {"choices": [{"delta": {"content": t}, "index": 0}], "model": model}
        chunks.append(f"data: {json.dumps(obj)}\n\n".encode())
    chunks.append(b"data: [DONE]\n\n")
    return chunks


# -- SSE accumulator --------------------------------------------------------

def test_accumulator_reassembles_deltas():
    acc = _SSEAccumulator()
    for c in _sse_chunks(["Hello", ", ", "world", "."]):
        acc.feed(c)
    assert acc.text == "Hello, world."


def test_accumulator_handles_split_across_chunk_boundaries():
    acc = _SSEAccumulator()
    # feed a single event split across two byte chunks
    full = _sse_chunks(["spanned"])[0]
    acc.feed(full[:6])
    acc.feed(full[6:])
    acc.feed(b"data: [DONE]\n\n")
    assert acc.text == "spanned"


def test_accumulator_ignores_unparseable_and_done():
    acc = _SSEAccumulator()
    acc.feed(b"data: not-json\n\n")
    acc.feed(b"data: [DONE]\n\n")
    assert acc.text == ""


# -- complete_stream --------------------------------------------------------

def _stream_proxy(engine, capture=None, texts=("You use ", "Kotlin.")):
    def stream_forward(payload):
        if capture is not None:
            capture["payload"] = payload
        return iter(_sse_chunks(list(texts)))
    return MemoryChatProxy(engine, ProxyConfig(), stream_forward_fn=stream_forward)


def test_complete_stream_passes_chunks_through(engine):
    proxy = _stream_proxy(engine)
    out = list(proxy.complete_stream(
        {"model": "gpt", "messages": [{"role": "user", "content": "hi"}]}, "s1"))
    joined = b"".join(out).decode()
    assert "You use " in joined and "Kotlin." in joined
    assert "[DONE]" in joined  # terminal event forwarded unchanged


def test_complete_stream_sets_stream_flag_and_injects(engine):
    engine.ingest_message("My name is Ada. I work at Analytical Engines.",
                          namespace="s2")
    capture = {}
    proxy = _stream_proxy(engine, capture=capture)
    list(proxy.complete_stream(
        {"model": "gpt", "messages": [
            {"role": "user", "content": "where do I work?"}]}, "s2"))
    fwd = capture["payload"]
    assert fwd["stream"] is True
    assert any(m["role"] == "system" and "analytical engines" in m["content"].lower()
               for m in fwd["messages"])


def test_complete_stream_records_assembled_reply(engine):
    proxy = _stream_proxy(engine, texts=("The project is named ", "DarkMatter."))
    list(proxy.complete_stream(
        {"model": "gpt", "messages": [{"role": "user", "content": "hi"}]}, "s3"))
    results = engine.retrieve("project name", namespace="s3")
    assert any("darkmatter" in r.text.lower() for r in results)


def test_complete_stream_records_user_turn(engine):
    proxy = _stream_proxy(engine)
    list(proxy.complete_stream(
        {"model": "gpt", "messages": [
            {"role": "user", "content": "My name is Bob and I use Kotlin."}]}, "s4"))
    results = engine.retrieve("what language does the user use?", namespace="s4")
    assert any("kotlin" in r.text.lower() for r in results)


# -- ASGI streaming ---------------------------------------------------------

def _call_asgi_stream(app, path, body: bytes, headers: dict):
    scope = {"type": "http", "method": "POST", "path": path,
             "headers": [[k.encode(), v.encode()] for k, v in headers.items()]}
    sent = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(ev):
        sent.append(ev)

    asyncio.run(app(scope, receive, send))
    start = next(e for e in sent if e["type"] == "http.response.start")
    body_out = b"".join(e.get("body", b"") for e in sent
                        if e["type"] == "http.response.body")
    ctype = dict((k.decode(), v.decode()) for k, v in start["headers"])
    return start["status"], ctype.get("content-type"), body_out


def test_asgi_streaming_passthrough_and_record(engine):
    proxy = _stream_proxy(engine, texts=("I use ", "Neovim."))
    app = create_app(proxy=proxy)
    body = json.dumps({"model": "gpt", "stream": True,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    status, ctype, out = _call_asgi_stream(
        app, "/v1/chat/completions", body, {"x-dmem-namespace": "s5"})
    assert status == 200
    assert ctype == "text/event-stream"
    assert b"Neovim." in out and b"[DONE]" in out
    # assembled reply recorded into the header-selected namespace
    assert any("neovim" in r.text.lower()
               for r in engine.retrieve("what does the user use?", namespace="s5"))
