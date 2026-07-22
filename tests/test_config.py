import pytest

from dmem import Config, Tier
from dmem.errors import ConfigError

# A complete set of pro-tier prerequisites for tests.
_PRO_ENV = {
    "PGVECTOR_URL": "postgresql://localhost/dmem",
    "GRAPH_DB": "neo4j",
    "GRAPH_DB_URL": "bolt://localhost:7687",
    "EMBEDDING_PROVIDER": "openai",
    "EMBEDDING_HOST_URL": "https://api.openai.com/v1",
    "EMBEDDING_API_KEY": "sk-test",
}


def test_default_tier_is_pro():
    cfg = Config.from_env(env=dict(_PRO_ENV))  # no MEMORY_TIER set
    assert cfg.tier is Tier.PRO


def test_empty_env_fails_fast():
    # pro is the default and its prerequisites are mandatory
    with pytest.raises(ConfigError) as e:
        Config.from_env(env={})
    msg = str(e.value)
    assert "GRAPH_DB_URL" in msg
    assert "PGVECTOR_URL" in msg
    assert "mandatory" in msg.lower()


def test_invalid_tier_raises():
    with pytest.raises(ConfigError):
        Config.from_env(env={"MEMORY_TIER": "quantum"})


def test_pro_requires_all_three_prerequisites():
    # missing embedding
    with pytest.raises(ConfigError) as e:
        Config.from_env(env={
            "PGVECTOR_URL": "postgresql://localhost/dmem",
            "GRAPH_DB": "neo4j", "GRAPH_DB_URL": "bolt://localhost:7687",
        })
    assert "EMBEDDING" in str(e.value)

    # missing graph
    with pytest.raises(ConfigError):
        Config.from_env(env={
            "PGVECTOR_URL": "postgresql://localhost/dmem",
            "EMBEDDING_HOST_URL": "https://api.openai.com/v1",
        })

    # missing pgvector
    with pytest.raises(ConfigError):
        Config.from_env(env={
            "GRAPH_DB": "neo4j", "GRAPH_DB_URL": "bolt://localhost:7687",
            "EMBEDDING_HOST_URL": "https://api.openai.com/v1",
        })


def test_pro_ok_with_all_prerequisites():
    cfg = Config.from_env(env=dict(_PRO_ENV))
    assert cfg.tier is Tier.PRO
    assert cfg.pgvector_url and cfg.graph.url and cfg.embedding.is_remote


def test_native_embedding_provider_counts_as_configured():
    env = dict(_PRO_ENV)
    del env["EMBEDDING_HOST_URL"]
    env["EMBEDDING_PROVIDER"] = "google"  # native provider, no host URL needed
    cfg = Config.from_env(env=env)
    assert cfg.embedding.is_remote


def test_sqlite_is_dev_tier_and_needs_nothing():
    cfg = Config.from_env(env={"MEMORY_TIER": "sqlite"})
    assert cfg.tier is Tier.SQLITE
    assert cfg.tier.is_dev is True


def test_consolidated_requires_graph():
    with pytest.raises(ConfigError):
        Config.from_env(env={"MEMORY_TIER": "consolidated"})


def test_invalid_graph_kind():
    with pytest.raises(ConfigError):
        Config.from_env(env={
            "MEMORY_TIER": "consolidated",
            "GRAPH_DB": "sqlserver",
            "GRAPH_DB_URL": "bolt://x",
        })


def test_offline_embeddings_flag():
    cfg = Config.from_env(env={"MEMORY_TIER": "sqlite"})
    assert cfg.uses_offline_embeddings is True
    cfg2 = Config.from_env(env={"MEMORY_TIER": "sqlite",
                                "EMBEDDING_HOST_URL": "http://localhost/v1"})
    assert cfg2.uses_offline_embeddings is False
