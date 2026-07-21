"""Exception hierarchy for DMem.

Kept deliberately small. The most important one is ``ConfigError`` — the brief
requires that a misconfigured tier fails *loudly at startup* rather than
silently downgrading to a weaker tier.
"""

from __future__ import annotations


class DMemError(Exception):
    """Base class for every error raised by DMem."""


class ConfigError(DMemError):
    """Raised when configuration is invalid or incomplete for the chosen tier.

    Per the brief: if ``tier=pro`` and a required backend URL is missing, we
    raise this at startup instead of silently falling back to a weaker tier.
    """


class ProviderError(DMemError):
    """Raised when a pluggable provider (embedding, OCR, reranker, LLM) fails."""


class StoreError(DMemError):
    """Raised when a backing store (SQLite / graph / pgvector) fails."""


class RetrievalError(DMemError):
    """Raised when retrieval cannot be completed."""
