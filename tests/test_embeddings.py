"""Embedding provider selection + per-vendor request/response shape.

Vendor adapters need the 'http' extra; the shape tests are guarded with
importorskip so the base offline suite is unaffected. Network is never touched —
only the pure _payload/_parse builders are exercised."""

import math

import pytest

from dmem.config import EmbeddingConfig
from dmem.errors import ProviderError
from dmem.providers.embeddings import (HashingEmbeddingProvider,
                                       HTTPEmbeddingProvider,
                                       build_embedding_provider)


def _unit(vec):
    return abs(math.sqrt(sum(x * x for x in vec)) - 1.0) < 1e-6


# -- selection --------------------------------------------------------------

def test_auto_offline_when_nothing_configured():
    p = build_embedding_provider(EmbeddingConfig())
    assert isinstance(p, HashingEmbeddingProvider)


def test_explicit_offline():
    p = build_embedding_provider(EmbeddingConfig(provider="offline"))
    assert isinstance(p, HashingEmbeddingProvider)


def test_openai_compatible_requires_host():
    with pytest.raises(ProviderError):
        build_embedding_provider(EmbeddingConfig(provider="openai"))


def test_openai_compatible_with_host(monkeypatch):
    pytest.importorskip("httpx")
    p = build_embedding_provider(EmbeddingConfig(
        provider="qwen", host_url="https://dashscope.example/compatible-mode/v1",
        model="text-embedding-v3", api_key="k"))
    assert isinstance(p, HTTPEmbeddingProvider)


def test_unknown_provider_raises():
    with pytest.raises(ProviderError):
        build_embedding_provider(EmbeddingConfig(provider="martian-embeddings"))


def test_signatures_are_vendor_distinct():
    pytest.importorskip("httpx")
    g = build_embedding_provider(EmbeddingConfig(provider="google",
                                                 model="text-embedding-004",
                                                 api_key="k"))
    c = build_embedding_provider(EmbeddingConfig(provider="cohere",
                                                 model="embed-v4.0", api_key="k"))
    assert g.signature().startswith("google:")
    assert c.signature().startswith("cohere:")
    assert g.signature() != c.signature()


# -- Google (Gemini) shape --------------------------------------------------

def test_google_payload_and_parse():
    pytest.importorskip("httpx")
    from dmem.providers.embeddings import GoogleEmbeddingProvider
    p = GoogleEmbeddingProvider(EmbeddingConfig(provider="google",
                                                model="text-embedding-004",
                                                api_key="secret"))
    payload = p._payload(["hello", "world"])
    assert payload["requests"][0]["model"] == "models/text-embedding-004"
    assert payload["requests"][1]["content"]["parts"][0]["text"] == "world"
    assert "key=secret" in p._endpoint()

    vecs = p._parse({"embeddings": [{"values": [3.0, 4.0]},
                                    {"values": [0.0, 5.0]}]})
    assert len(vecs) == 2
    assert _unit(vecs[0])  # normalized


def test_google_requires_api_key():
    pytest.importorskip("httpx")
    from dmem.providers.embeddings import GoogleEmbeddingProvider
    with pytest.raises(ProviderError):
        GoogleEmbeddingProvider(EmbeddingConfig(provider="google"))


# -- Cohere shape -----------------------------------------------------------

def test_cohere_payload_and_parse():
    pytest.importorskip("httpx")
    from dmem.providers.embeddings import CohereEmbeddingProvider
    p = CohereEmbeddingProvider(EmbeddingConfig(provider="cohere",
                                                model="embed-v4.0", api_key="k"))
    payload = p._payload(["a", "b"])
    assert payload["texts"] == ["a", "b"]
    assert payload["input_type"] == "search_document"
    assert payload["embedding_types"] == ["float"]
    assert p._url.endswith("/embed")

    # v2 nested shape
    vecs = p._parse({"embeddings": {"float": [[3.0, 4.0], [1.0, 0.0]]}})
    assert len(vecs) == 2 and _unit(vecs[0])
    # tolerant of a flat list shape too
    assert len(p._parse({"embeddings": [[1.0, 0.0]]})) == 1


# -- pgvector coexistence prefix --------------------------------------------

def test_pgvector_table_prefix_isolates():
    pytest.importorskip("psycopg")
    from dmem.stores.pgvector_store import PgVectorStore
    store = PgVectorStore("postgresql://x/y", table_prefix="myapp")
    assert store._chunks == "myapp_chunks"
    assert store._meta == "myapp_meta"


def test_pgvector_rejects_bad_prefix():
    pytest.importorskip("psycopg")
    from dmem.errors import StoreError
    from dmem.stores.pgvector_store import PgVectorStore
    with pytest.raises(StoreError):
        PgVectorStore("postgresql://x/y", table_prefix="bad-prefix; DROP")
