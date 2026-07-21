"""Environment-variable configuration with fail-fast tier validation.

Design rules from the brief:
- Adopter picks the tier explicitly via ``MEMORY_TIER`` — never auto-detected.
- Everything is bring-your-own-endpoint. Nothing about a specific provider is
  hardcoded; the only defaults are structural (tier selection, budgets).
- If a required backend for the chosen tier is missing, raise ``ConfigError``
  at startup. **Never silently downgrade to a weaker tier.**

Config is intentionally sourced from env vars only (no wizard, no config file
parsing) so the same code path works identically across the SDK, the MCP
server, and the IDE extension.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .errors import ConfigError
from .types import Cardinality


class Tier(str, Enum):
    SQLITE = "sqlite"              # zero-setup default, no server
    CONSOLIDATED = "consolidated"  # one graph DB w/ native vectors
    PRO = "pro"                   # pgvector + dedicated Graphiti-backed graph

    @classmethod
    def coerce(cls, value: Optional[str]) -> "Tier":
        if not value:
            return cls.SQLITE
        try:
            return cls(value.strip().lower())
        except ValueError as e:
            raise ConfigError(
                f"MEMORY_TIER={value!r} is invalid. "
                f"Choose one of: {', '.join(t.value for t in cls)}."
            ) from e


class GraphKind(str, Enum):
    NEO4J = "neo4j"
    FALKORDB = "falkordb"

    @classmethod
    def coerce(cls, value: Optional[str]) -> Optional["GraphKind"]:
        if not value:
            return None
        try:
            return cls(value.strip().lower())
        except ValueError as e:
            raise ConfigError(
                f"GRAPH_DB={value!r} is invalid. Choose 'neo4j' or 'falkordb'."
            ) from e


class EmbeddingMismatchPolicy(str, Enum):
    """What to do when the store was built with a different embedding model.

    Vectors from different embedding models are not comparable, so a silent
    model swap corrupts retrieval. Default is to warn loudly."""

    WARN = "warn"     # log a warning, keep going (default)
    ERROR = "error"   # refuse to start until re-embedded or reverted
    IGNORE = "ignore"  # say nothing (not recommended)

    @classmethod
    def coerce(cls, value: Optional[str]) -> "EmbeddingMismatchPolicy":
        if not value:
            return cls.WARN
        try:
            return cls(value.strip().lower())
        except ValueError as e:
            raise ConfigError(
                f"EMBEDDING_MISMATCH_POLICY={value!r} is invalid. "
                "Choose warn|error|ignore.") from e


class CredentialPolicy(str, Enum):
    """How to handle facts extracted as credentials (secret-bearing).

    - redact (default): store the fact but mask the secret value and never embed
      it — the system knows a credential was mentioned without retaining it.
    - drop: do not store credential facts at all.
    - store: store the raw value (still withheld from handoffs). Opt-in only."""

    REDACT = "redact"
    DROP = "drop"
    STORE = "store"

    @classmethod
    def coerce(cls, value: Optional[str]) -> "CredentialPolicy":
        if not value:
            return cls.REDACT
        try:
            return cls(value.strip().lower())
        except ValueError as e:
            raise ConfigError(
                f"CREDENTIAL_POLICY={value!r} is invalid. "
                "Choose redact|drop|store.") from e


def _get(env: dict[str, str], *keys: str) -> Optional[str]:
    for k in keys:
        v = env.get(k)
        if v is not None and v.strip() != "":
            return v.strip()
    return None


def _get_int(env: dict[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ConfigError(f"{key}={raw!r} must be an integer.") from e


def _parse_cardinality_overrides(env: dict[str, str]) -> dict[str, Cardinality]:
    """Build predicate->cardinality overrides from two comma-separated env vars."""
    overrides: dict[str, Cardinality] = {}
    for pred in (env.get("MULTI_VALUED_PREDICATES") or "").split(","):
        p = pred.strip().lower()
        if p:
            overrides[p] = Cardinality.MULTI
    for pred in (env.get("SINGLE_VALUED_PREDICATES") or "").split(","):
        p = pred.strip().lower()
        if p:
            overrides[p] = Cardinality.SINGLE
    return overrides


def _get_float(env: dict[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ConfigError(f"{key}={raw!r} must be a number.") from e


@dataclass
class EmbeddingConfig:
    # provider selects the API *shape*: "openai" (OpenAI-compatible — covers
    # Qwen/DashScope, vLLM, TEI, Ollama, Azure, Together, ...), "google" (Gemini),
    # "cohere", or "offline". Empty => auto: openai-compatible if a host_url is
    # set, else the offline dev embedder.
    provider: Optional[str] = None
    model: Optional[str] = None
    host_url: Optional[str] = None
    api_key: Optional[str] = None
    dim: Optional[int] = None  # optional hint; providers can report their own

    @property
    def is_remote(self) -> bool:
        return bool(self.host_url) or bool(self.provider and
                                           self.provider.lower() in _REMOTE_PROVIDERS)


# Providers that reach a remote endpoint even without an explicit host_url
# (they have a well-known default base URL).
_REMOTE_PROVIDERS = {"google", "gemini", "cohere"}


@dataclass
class OCRConfig:
    host_url: Optional[str] = None
    api_key: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.host_url)


@dataclass
class GraphConfig:
    kind: Optional[GraphKind] = None
    url: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None


@dataclass
class LLMConfig:
    """Optional LLM used for high-quality fact extraction / summarization.

    Bring-your-own. When absent, the engine uses the deterministic heuristic
    extractor (lower quality, but zero-setup and offline)."""

    model: Optional[str] = None
    host_url: Optional[str] = None
    api_key: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.host_url)


@dataclass
class Config:
    tier: Tier = Tier.SQLITE

    # SQLite tier
    sqlite_path: str = "dmem.db"

    # Pluggable providers
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Pro-tier document store
    pgvector_url: Optional[str] = None
    # Table/index name prefix — so DMem coexists in an existing Postgres without
    # colliding with your tables. Change it to run multiple DMem instances in one DB.
    pgvector_table_prefix: str = "dmem"

    # Graph backend (consolidated + pro)
    graph: GraphConfig = field(default_factory=GraphConfig)

    # Retrieval / handoff tuning (safe day-one defaults; benchmarking deferred)
    handoff_token_budget: int = 1200
    retrieval_top_k: int = 20
    rerank_top_k: int = 8
    recency_half_life_days: float = 30.0
    low_confidence_threshold: float = 0.15  # below => hard fallback
    rerank_enabled: bool = True

    # Reranker (optional cross-encoder). BYO model name; local by default.
    rerank_model: Optional[str] = None

    default_namespace: str = "default"

    # Production-safety policies.
    embedding_mismatch_policy: EmbeddingMismatchPolicy = EmbeddingMismatchPolicy.WARN
    credential_policy: CredentialPolicy = CredentialPolicy.REDACT

    # Per-predicate cardinality overrides (extends the built-in defaults).
    # Controls contradiction handling: SINGLE supersedes on change, MULTI
    # accumulates. Populated from MULTI_VALUED_PREDICATES / SINGLE_VALUED_PREDICATES.
    predicate_cardinality_overrides: dict[str, Cardinality] = field(default_factory=dict)

    # --------------------------------------------------------------------- #
    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None) -> "Config":
        env = dict(os.environ if env is None else env)

        tier = Tier.coerce(env.get("MEMORY_TIER"))

        cfg = cls(
            tier=tier,
            sqlite_path=_get(env, "SQLITE_PATH", "DMEM_SQLITE_PATH") or "dmem.db",
            embedding=EmbeddingConfig(
                provider=_get(env, "EMBEDDING_PROVIDER"),
                model=_get(env, "EMBEDDING_MODEL"),
                host_url=_get(env, "EMBEDDING_HOST_URL"),
                api_key=_get(env, "EMBEDDING_API_KEY"),
                dim=_get_int(env, "EMBEDDING_DIM", 0) or None,
            ),
            ocr=OCRConfig(
                host_url=_get(env, "OCR_HOST_URL"),
                api_key=_get(env, "OCR_API_KEY"),
            ),
            llm=LLMConfig(
                model=_get(env, "LLM_MODEL"),
                host_url=_get(env, "LLM_HOST_URL"),
                api_key=_get(env, "LLM_API_KEY"),
            ),
            pgvector_url=_get(env, "PGVECTOR_URL"),
            pgvector_table_prefix=_get(env, "PGVECTOR_TABLE_PREFIX") or "dmem",
            graph=GraphConfig(
                kind=GraphKind.coerce(env.get("GRAPH_DB")),
                url=_get(env, "GRAPH_DB_URL"),
                user=_get(env, "GRAPH_DB_USER"),
                password=_get(env, "GRAPH_DB_PASSWORD"),
                database=_get(env, "GRAPH_DB_DATABASE"),
            ),
            handoff_token_budget=_get_int(env, "HANDOFF_TOKEN_BUDGET", 1200),
            retrieval_top_k=_get_int(env, "RETRIEVAL_TOP_K", 20),
            rerank_top_k=_get_int(env, "RERANK_TOP_K", 8),
            recency_half_life_days=_get_float(env, "RECENCY_HALF_LIFE_DAYS", 30.0),
            low_confidence_threshold=_get_float(env, "LOW_CONFIDENCE_THRESHOLD", 0.15),
            rerank_enabled=(_get(env, "RERANK_ENABLED") or "true").lower()
            not in ("0", "false", "no"),
            rerank_model=_get(env, "RERANK_MODEL"),
            default_namespace=_get(env, "DMEM_NAMESPACE") or "default",
            embedding_mismatch_policy=EmbeddingMismatchPolicy.coerce(
                env.get("EMBEDDING_MISMATCH_POLICY")),
            credential_policy=CredentialPolicy.coerce(env.get("CREDENTIAL_POLICY")),
            predicate_cardinality_overrides=_parse_cardinality_overrides(env),
        )
        cfg.validate()
        return cfg

    # --------------------------------------------------------------------- #
    def validate(self) -> None:
        """Fail fast for the chosen tier. Never silently downgrade."""
        missing: list[str] = []

        if self.tier is Tier.PRO:
            if not self.pgvector_url:
                missing.append("PGVECTOR_URL (required for tier=pro)")
            if not self.graph.url:
                missing.append("GRAPH_DB_URL (required for tier=pro)")
            if not self.graph.kind:
                missing.append("GRAPH_DB (neo4j|falkordb, required for tier=pro)")

        elif self.tier is Tier.CONSOLIDATED:
            if not self.graph.url:
                missing.append("GRAPH_DB_URL (required for tier=consolidated)")
            if not self.graph.kind:
                missing.append(
                    "GRAPH_DB (neo4j|falkordb, required for tier=consolidated)"
                )

        if missing:
            raise ConfigError(
                f"tier={self.tier.value} is missing required configuration:\n  - "
                + "\n  - ".join(missing)
                + "\n\nSet these environment variables, or choose a lower tier "
                "explicitly with MEMORY_TIER=sqlite. DMem will not silently "
                "downgrade."
            )

        if self.handoff_token_budget <= 0:
            raise ConfigError("HANDOFF_TOKEN_BUDGET must be > 0.")
        if self.recency_half_life_days <= 0:
            raise ConfigError("RECENCY_HALF_LIFE_DAYS must be > 0.")

    # Convenience -----------------------------------------------------------
    @property
    def uses_offline_embeddings(self) -> bool:
        """True when no remote embedding endpoint is configured.

        The engine will use the deterministic offline embedder and emit a
        warning. This keeps the SQLite tier genuinely zero-setup while making
        it obvious that retrieval quality is degraded."""
        return not self.embedding.is_remote
