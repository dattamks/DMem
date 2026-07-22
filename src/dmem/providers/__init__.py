"""Pluggable providers: embeddings, reranking, OCR, and (optional) LLM.

Every provider is bring-your-own-endpoint. Nothing about a specific vendor is
hardcoded. When a remote endpoint is not configured, a deterministic offline
fallback is used so the zero-setup SQLite tier still runs end-to-end (with a
clear warning that quality is degraded)."""

from .embeddings import EmbeddingProvider, build_embedding_provider
from .reranker import Reranker, build_reranker
from .ocr import OCRProvider, build_ocr_provider
from .llm import LLMProvider, build_llm_provider

__all__ = [
    "EmbeddingProvider",
    "build_embedding_provider",
    "Reranker",
    "build_reranker",
    "OCRProvider",
    "build_ocr_provider",
    "LLMProvider",
    "build_llm_provider",
]
