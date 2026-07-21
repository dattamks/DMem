"""Embedding-model-change guard: switching models must not silently corrupt
retrieval. The store stamps the embedding signature; the engine checks it."""

import warnings

import pytest

from dmem import Config, DMemEngine, Tier
from dmem.config import EmbeddingConfig, EmbeddingMismatchPolicy
from dmem.errors import ConfigError


# Ignore the offline-embedder warning module-wide so it doesn't drown the
# mismatch-warning assertions; the guard warning is what these tests check.
warnings.filterwarnings("ignore", message=".*offline hashing embedder.*")


def _engine(path, dim=256, policy=EmbeddingMismatchPolicy.WARN):
    cfg = Config(tier=Tier.SQLITE, sqlite_path=str(path),
                 embedding=EmbeddingConfig(dim=dim),
                 embedding_mismatch_policy=policy)
    return DMemEngine(cfg)


def test_signature_stamped_on_first_open(tmp_path):
    e = _engine(tmp_path / "s.db")
    sig = e.store.get_meta("embedding_signature")
    assert sig == e.embedder.signature()
    assert e.store.get_meta("schema_version") == "1"
    e.close()


def test_same_model_reopens_cleanly(tmp_path):
    p = tmp_path / "s.db"
    e1 = _engine(p, dim=256)
    e1.ingest_message("My name is Ada.")
    e1.close()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e2 = _engine(p, dim=256)
    assert not any("embedding model changed" in str(x.message) for x in w)
    e2.close()


def test_model_change_warns(tmp_path):
    p = tmp_path / "s.db"
    e1 = _engine(p, dim=256)
    e1.close()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e2 = _engine(p, dim=512, policy=EmbeddingMismatchPolicy.WARN)  # different dim
    assert any("embedding model changed" in str(x.message).lower() for x in w)
    e2.close()


def test_model_change_error_policy_raises(tmp_path):
    p = tmp_path / "s.db"
    e1 = _engine(p, dim=256)
    e1.close()
    with pytest.raises(ConfigError):
        _engine(p, dim=512, policy=EmbeddingMismatchPolicy.ERROR)


def test_reembed_updates_vectors_and_signature(tmp_path):
    p = tmp_path / "s.db"
    e1 = _engine(p, dim=256)
    e1.ingest_message("My name is Ada.")
    e1.ingest_document("# Doc\n\nProd runs on Postgres.", document_id="d")
    e1.close()

    # reopen with a different embedder, re-embed, signature should update
    e2 = _engine(p, dim=512, policy=EmbeddingMismatchPolicy.WARN)
    n = e2.reembed()
    assert n >= 2  # at least the fact + the chunk
    assert e2.store.get_meta("embedding_signature") == e2.embedder.signature()
    # retrieval works again after re-embedding
    results = e2.retrieve("what is the user's name?")
    assert any("ada" in r.text.lower() for r in results)
    e2.close()
