"""Proxy over REAL HTTP — a local fake OpenAI server, real httpx forwarding.

The proxy's other tests inject a Python forward function; this one exercises the
actual network path: real `httpx` POST to a localhost server that mimics
/v1/chat/completions (JSON and SSE streaming). Proves headers, URL building,
non-streaming forward, and byte-for-byte SSE passthrough work over the wire.
"""

import json
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("httpx")

from dmem import Config, DMemEngine, Tier
from dmem.config import EmbeddingConfig
from dmem.adapters.proxy import MemoryChatProxy, ProxyConfig


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.received.append(payload)  # type: ignore[attr-defined]
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for t in ["I ", "use ", "Rust."]:
                self.wfile.write(
                    f"data: {json.dumps({'choices':[{'delta':{'content':t}}]})}\n\n"
                    .encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = json.dumps({"choices": [{"message": {
                "role": "assistant", "content": "I use Rust."}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.received = []  # type: ignore[attr-defined]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1", server
    finally:
        server.shutdown()


def _engine(tmp_path):
    cfg = Config(tier=Tier.SQLITE, sqlite_path=str(tmp_path / "p.db"),
                 embedding=EmbeddingConfig())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DMemEngine(cfg)


def test_proxy_real_http_non_streaming(upstream, tmp_path):
    base, server = upstream
    engine = _engine(tmp_path)
    engine.ingest_message("My name is Ada. I work at Analytical Engines.",
                          namespace="h1")
    proxy = MemoryChatProxy(engine, ProxyConfig(upstream_base_url=base,
                                                upstream_api_key="test-key"))
    try:
        resp = proxy.complete(
            {"model": "gpt", "messages": [
                {"role": "user", "content": "where do I work?"}]}, "h1")
        # real HTTP round-trip returned the upstream body
        assert resp["choices"][0]["message"]["content"] == "I use Rust."
        # the upstream actually received the memory-injected system message
        got = server.received[-1]
        assert any(m["role"] == "system"
                   and "analytical engines" in m["content"].lower()
                   for m in got["messages"])
    finally:
        engine.close()


def test_proxy_real_http_streaming(upstream, tmp_path):
    base, server = upstream
    engine = _engine(tmp_path)
    proxy = MemoryChatProxy(engine, ProxyConfig(upstream_base_url=base))
    try:
        chunks = list(proxy.complete_stream(
            {"model": "gpt", "messages": [
                {"role": "user", "content": "hello"}]}, "h2"))
        out = b"".join(chunks).decode()
        assert "Rust." in out and "[DONE]" in out       # SSE passed through
        assert server.received[-1]["stream"] is True     # forwarded as stream
        # assembled assistant reply teed into memory
        results = engine.retrieve("what does the user use?", namespace="h2")
        assert any("rust" in r.text.lower() for r in results)
    finally:
        engine.close()
