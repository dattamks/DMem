"""DMem — pluggable, self-hostable AI memory & context library.

Persistent memory + document RAG + model-switch continuity, distributed as a
library (not a hosted service). One core engine; thin adapters for the Python
SDK, an MCP server, and an IDE extension.

Quick start:

    from dmem import DMemEngine

    mem = DMemEngine()                       # reads config from env (MEMORY_TIER=...)
    mem.ingest_message("I prefer dark mode and I use Postgres.")
    handoff = mem.handoff("what are the user's preferences?")
    print(handoff.text)
"""

from .config import Config, Tier
from .engine import DMemEngine
from .errors import ConfigError, DMemError, ProviderError, RetrievalError, StoreError
from .types import (Chunk, Episode, Fact, FactType, Handoff, Provenance,
                    RetrievalResult)

__version__ = "0.1.0"

__all__ = [
    "DMemEngine",
    "Config",
    "Tier",
    "Fact",
    "FactType",
    "Chunk",
    "Episode",
    "Provenance",
    "Handoff",
    "RetrievalResult",
    "DMemError",
    "ConfigError",
    "ProviderError",
    "StoreError",
    "RetrievalError",
]
