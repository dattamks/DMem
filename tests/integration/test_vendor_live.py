"""Real cloud-vendor smoke tests — the ONE thing localhost fakes can't verify:
that the vendor's actual response shape parses and the model works.

Every test skips unless its API key is present, so this file is safe to run
anywhere (it no-ops without secrets). In CI, the `vendor-live` job supplies keys
from repo secrets. Locally: export the key and run.
"""

import math
import os

import pytest

pytest.importorskip("httpx")

from dmem.config import EmbeddingConfig
from dmem.providers.embeddings import build_embedding_provider

OPENAI = os.environ.get("OPENAI_API_KEY")
GOOGLE = os.environ.get("GOOGLE_API_KEY")
COHERE = os.environ.get("COHERE_API_KEY")


def _normalized(v):
    return abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-2


@pytest.mark.skipif(not OPENAI, reason="set OPENAI_API_KEY")
def test_openai_embeddings_live():
    p = build_embedding_provider(EmbeddingConfig(
        provider="openai", host_url="https://api.openai.com/v1",
        model="text-embedding-3-small", api_key=OPENAI))
    v = p.embed_one("the quick brown fox")
    assert len(v) > 256 and _normalized(v)          # real model dimension
    # semantically related > unrelated (real embeddings, real signal)
    a, b, c = p.embed(["cat", "kitten", "airplane"])
    assert _dot(a, b) > _dot(a, c)


@pytest.mark.skipif(not GOOGLE, reason="set GOOGLE_API_KEY")
def test_google_embeddings_live():
    p = build_embedding_provider(EmbeddingConfig(
        provider="google", model="text-embedding-004", api_key=GOOGLE))
    v = p.embed_one("the quick brown fox")
    assert len(v) > 256 and _normalized(v)


@pytest.mark.skipif(not COHERE, reason="set COHERE_API_KEY")
def test_cohere_embeddings_live():
    p = build_embedding_provider(EmbeddingConfig(
        provider="cohere", model="embed-v4.0", api_key=COHERE))
    v = p.embed_one("the quick brown fox")
    assert len(v) > 256 and _normalized(v)


@pytest.mark.skipif(not OPENAI, reason="set OPENAI_API_KEY")
def test_retrieval_quality_with_real_embedder(tmp_path):
    """Turn the offline retrieval floor into a real quality number."""
    import warnings
    from dmem import Config, Tier
    from dmem.eval import run_smoke
    cfg = Config(tier=Tier.SQLITE, sqlite_path=str(tmp_path / "q.db"),
                 embedding=EmbeddingConfig(
                     provider="openai", host_url="https://api.openai.com/v1",
                     model="text-embedding-3-small", api_key=OPENAI))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = run_smoke(cfg)
    # a real embedder should comfortably clear the retrieval bar
    assert report.hit_at_k >= 0.8
    assert report.all_checks_passed


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))
