import pytest

from dmem import Config, Tier
from dmem.errors import ConfigError


def test_default_tier_is_sqlite():
    cfg = Config.from_env(env={})
    assert cfg.tier is Tier.SQLITE


def test_invalid_tier_raises():
    with pytest.raises(ConfigError):
        Config.from_env(env={"MEMORY_TIER": "quantum"})


def test_pro_tier_fails_fast_without_backends():
    with pytest.raises(ConfigError) as e:
        Config.from_env(env={"MEMORY_TIER": "pro"})
    msg = str(e.value)
    assert "PGVECTOR_URL" in msg
    assert "GRAPH_DB_URL" in msg
    # never silently downgrades
    assert "will not silently" in msg.lower()


def test_consolidated_requires_graph():
    with pytest.raises(ConfigError):
        Config.from_env(env={"MEMORY_TIER": "consolidated"})


def test_pro_tier_ok_with_backends():
    cfg = Config.from_env(env={
        "MEMORY_TIER": "pro",
        "PGVECTOR_URL": "postgresql://localhost/dmem",
        "GRAPH_DB": "neo4j",
        "GRAPH_DB_URL": "bolt://localhost:7687",
    })
    assert cfg.tier is Tier.PRO
    assert cfg.pgvector_url


def test_invalid_graph_kind():
    with pytest.raises(ConfigError):
        Config.from_env(env={
            "MEMORY_TIER": "consolidated",
            "GRAPH_DB": "sqlserver",
            "GRAPH_DB_URL": "bolt://x",
        })


def test_offline_embeddings_flag():
    cfg = Config.from_env(env={})
    assert cfg.uses_offline_embeddings is True
    cfg2 = Config.from_env(env={"EMBEDDING_HOST_URL": "http://localhost/v1"})
    assert cfg2.uses_offline_embeddings is False
