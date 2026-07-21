import warnings

import pytest

from dmem import Config, DMemEngine, Tier
from dmem.config import EmbeddingConfig


@pytest.fixture
def engine(tmp_path):
    cfg = Config(
        tier=Tier.SQLITE,
        sqlite_path=str(tmp_path / "test.db"),
        embedding=EmbeddingConfig(),  # offline hashing embedder
        rerank_enabled=True,          # lexical reranker (offline)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = DMemEngine(cfg)
    yield eng
    eng.close()
