"""Embedding providers over REAL HTTP — local fake vendor servers.

Validates the actual httpx calls (URL building, auth headers, request body, and
response parsing) for all three provider shapes against a localhost server that
mimics each vendor's API. No cloud calls; no network beyond loopback.
"""

import json
import threading
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("httpx")

from dmem.config import EmbeddingConfig
from dmem.providers.embeddings import (HTTPEmbeddingProvider,
                                       GoogleEmbeddingProvider,
                                       CohereEmbeddingProvider)


class _VendorHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.last = {  # type: ignore[attr-defined]
            "path": self.path, "auth": self.headers.get("Authorization"),
            "body": body}
        path = self.path.split("?")[0]

        if path.endswith("/embeddings"):          # OpenAI-compatible
            n = len(body.get("input", []))
            out = {"data": [{"embedding": [1.0, 2.0, 2.0], "index": i}
                            for i in range(n)]}
        elif path.endswith(":batchEmbedContents"):  # Google Gemini
            n = len(body.get("requests", []))
            out = {"embeddings": [{"values": [3.0, 4.0]} for _ in range(n)]}
        elif path.endswith("/embed"):              # Cohere v2
            n = len(body.get("texts", []))
            out = {"embeddings": {"float": [[0.0, 5.0] for _ in range(n)]}}
        else:
            self.send_response(404); self.end_headers(); return

        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def vendor():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _VendorHandler)
    server.last = None  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", server
    finally:
        server.shutdown()


def _unit(v):
    import math
    return abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-6


def test_openai_compatible_over_http(vendor):
    base, server = vendor
    p = HTTPEmbeddingProvider(EmbeddingConfig(
        provider="openai", host_url=f"{base}/v1", model="text-embedding-3-small",
        api_key="sk-abc"))
    vecs = p.embed(["hello", "world"])
    assert len(vecs) == 2 and _unit(vecs[0])          # normalized
    assert server.last["path"].endswith("/v1/embeddings")
    assert server.last["auth"] == "Bearer sk-abc"      # auth header sent
    assert server.last["body"]["model"] == "text-embedding-3-small"
    assert server.last["body"]["input"] == ["hello", "world"]


def test_google_over_http(vendor):
    base, server = vendor
    p = GoogleEmbeddingProvider(EmbeddingConfig(
        provider="google", host_url=base, model="text-embedding-004",
        api_key="gkey"))
    vecs = p.embed(["a", "b"])
    assert len(vecs) == 2 and _unit(vecs[0])
    assert ":batchEmbedContents" in server.last["path"]
    assert "key=gkey" in server.last["path"]           # key in query string
    assert server.last["body"]["requests"][0]["model"] == "models/text-embedding-004"


def test_cohere_over_http(vendor):
    base, server = vendor
    p = CohereEmbeddingProvider(EmbeddingConfig(
        provider="cohere", host_url=base, model="embed-v4.0", api_key="ck"))
    vecs = p.embed(["x", "y"])
    assert len(vecs) == 2 and _unit(vecs[0])
    assert server.last["path"].endswith("/embed")
    assert server.last["auth"] == "Bearer ck"
    assert server.last["body"]["input_type"] == "search_document"
    assert server.last["body"]["embedding_types"] == ["float"]


def test_embed_one_and_dim_discovery(vendor):
    base, server = vendor
    p = HTTPEmbeddingProvider(EmbeddingConfig(
        provider="openai", host_url=f"{base}/v1", model="m", api_key="k"))
    v = p.embed_one("solo")
    assert len(v) == 3
    assert p.dim == 3   # discovered from the first response
