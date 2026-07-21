"""DMemEngine — the single shared core every distribution surface wraps.

The SDK, the MCP server, and the IDE extension are all thin adapters over this
class. No memory logic is duplicated per surface.

Public surface:
- ``ingest_message(text, ...)``      remember a conversational turn
- ``ingest_document(data, ...)``     OCR -> OKF -> chunk -> index a document
- ``retrieve(query, ...)``           hybrid retrieval (the escape-hatch tool)
- ``handoff(query, ...)``            token-budgeted model-switch context
- ``compact(transcript, ...)``       shrink a long same-model transcript
- ``forget(namespace)``              hard-delete a namespace (GDPR)

Failure posture: for a library embedded in someone else's app, retrieval failure
should degrade gracefully. ``retrieve`` and ``handoff`` never raise on empty
results — they return an explicit low-confidence signal so the caller can invoke
the escape hatch or surface the gap, per the "hard fallback" requirement.
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

from .config import Config
from .pipeline.chunking import chunk_concepts
from .pipeline.dedup import content_hash
from .pipeline.extraction import (FactExtractor, HeuristicFactExtractor,
                                  LLMFactExtractor)
from .pipeline.okf import structure_markdown
from .providers.embeddings import build_embedding_provider
from .providers.llm import build_llm_provider
from .providers.ocr import build_ocr_provider
from .providers.reranker import build_reranker
from .retrieval.handoff import build_handoff
from .retrieval.hybrid import HybridRetriever
from .retrieval.scoring import detect_conflicts, reciprocal_rank_fusion
from .stores.factory import build_store
from .types import (Cardinality, Chunk, Episode, Fact, Handoff, Provenance,
                    RetrievalResult, predicate_cardinality)


class DMemEngine:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        cfg = self.config

        self.embedder = build_embedding_provider(cfg.embedding)
        embed_dim = getattr(self.embedder, "dim", 256) or 256

        self.store = build_store(cfg, embed_dim=embed_dim)
        self.store.initialize()

        self.reranker = build_reranker(cfg.rerank_enabled, cfg.rerank_model)
        self.ocr = build_ocr_provider(cfg.ocr)
        self.llm = build_llm_provider(cfg.llm)

        self.extractor: FactExtractor = (
            LLMFactExtractor(self.llm) if self.llm is not None
            else HeuristicFactExtractor())

        self.retriever = HybridRetriever(
            self.store, self.embedder, self.reranker,
            top_k=cfg.retrieval_top_k, rerank_top_k=cfg.rerank_top_k,
            recency_half_life_days=cfg.recency_half_life_days)

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_message(self, text: str, *, namespace: Optional[str] = None,
                       role: str = "user", speaker: Optional[str] = None,
                       origin_id: Optional[str] = None) -> list[Fact]:
        """Extract typed facts from a message and store them, handling
        contradiction (supersede, don't delete) and dedup."""
        ns = namespace or self.config.default_namespace
        ep = Episode(text=text, namespace=ns, role=role)
        speaker_label = speaker or (role if role != "user" else "user")

        extracted = self.extractor.extract(text, speaker=speaker_label)
        stored: list[Fact] = []
        now = time.time()
        for ex in extracted:
            prov = Provenance(source="conversation", timestamp=now,
                              origin_id=origin_id or ep.id, actor=speaker_label)
            fact = Fact(subject=ex.subject, predicate=ex.predicate,
                        object=ex.object, fact_type=ex.fact_type,
                        provenance=prov, namespace=ns, confidence=ex.confidence,
                        valid_from=now, recorded_at=now)
            supersede = self._should_supersede(ex)
            saved = self._store_fact_with_conflict_handling(fact, now, supersede)
            if saved is not None:
                stored.append(saved)
        return stored

    def _should_supersede(self, ex) -> bool:
        """Decide whether a new value retires prior values of the same
        (subject, predicate), based on the predicate's cardinality.

        - Explicit ``replaces`` from the extractor wins (True/False).
        - Otherwise SINGLE-valued predicates supersede; MULTI-valued accumulate.
        A cardinality hint on the extracted fact overrides the registry."""
        if getattr(ex, "replaces", None) is True:
            return True
        if getattr(ex, "replaces", None) is False:
            return False
        card = getattr(ex, "cardinality", None) or predicate_cardinality(
            ex.predicate, self.config.predicate_cardinality_overrides)
        return card is Cardinality.SINGLE

    def _store_fact_with_conflict_handling(
        self, fact: Fact, now: float, supersede: bool = True) -> Optional[Fact]:
        # 1) Exact dedup: identical subject+predicate+object already current.
        existing = self.store.find_fact_by_dedup_key(fact.namespace,
                                                     fact.dedup_key())
        if existing is not None:
            return None  # already known; skip (dedup)

        # 2) Contradiction handling — only for single-valued (or explicitly
        #    replacing) predicates. Multi-valued predicates accumulate:
        #    "I use Postgres" and "I use Redis" are both current, not a conflict.
        if supersede:
            current = self.store.find_current_facts(fact.namespace, fact.subject,
                                                    fact.predicate)
            superseded_id = None
            for old in current:
                if old.object.strip().lower() != fact.object.strip().lower():
                    self.store.close_fact(old.id, valid_to=now)
                    superseded_id = old.id  # link the most recent one
            if superseded_id:
                fact.supersedes = superseded_id

        fact.embedding = self.embedder.embed_one(fact.statement)
        self.store.upsert_fact(fact)
        return fact

    def ingest_document(self, data, *, filename: Optional[str] = None,
                        document_id: Optional[str] = None,
                        namespace: Optional[str] = None,
                        media_type: Optional[str] = None) -> list[Chunk]:
        """OCR (if needed) -> OKF-structure -> chunk -> embed -> index.

        Accepts markdown/text ``str`` directly, or ``bytes`` for binary docs
        (routed through the configured OCR endpoint). Dedups by chunk content
        hash so re-ingesting the same document doesn't bloat the store."""
        ns = namespace or self.config.default_namespace
        doc_id = document_id or (filename or "document")

        markdown = self._to_markdown(data, filename, media_type)
        concepts = structure_markdown(markdown, default_title=doc_id)
        prov = Provenance(source="document", timestamp=time.time(),
                          origin_id=doc_id, origin_ref=filename)
        chunks = chunk_concepts(concepts, doc_id, namespace=ns, provenance=prov)

        fresh: list[Chunk] = []
        for c in chunks:
            if self.store.chunk_exists(ns, content_hash(c.text)):
                continue  # dedup
            c.embedding = self.embedder.embed_one(c.text)
            fresh.append(c)
        if fresh:
            self.store.upsert_chunks(fresh)
        return fresh

    def _to_markdown(self, data, filename, media_type) -> str:
        if isinstance(data, str):
            return data
        if isinstance(data, (bytes, bytearray)):
            # Text-ish bytes: decode directly. Otherwise require OCR.
            if _looks_texty(media_type, filename):
                return bytes(data).decode("utf-8", errors="replace")
            if self.ocr is None:
                raise ValueError(
                    "Binary document submitted but no OCR endpoint configured. "
                    "Set OCR_HOST_URL, or pass already-extracted text/markdown.")
            return self.ocr.to_markdown(bytes(data), filename)
        raise TypeError("ingest_document expects str or bytes.")

    # ------------------------------------------------------------------ #
    # Retrieval / escape hatch
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, *, namespace: Optional[str] = None,
                 kinds: tuple[str, ...] = ("fact", "chunk"),
                 seeds: Optional[list[str]] = None) -> list[RetrievalResult]:
        """Hybrid retrieval. This IS the escape-hatch tool the model calls
        mid-conversation when it senses a context gap. Never raises on empty."""
        ns = namespace or self.config.default_namespace
        try:
            return self.retriever.retrieve(ns, query, kinds=kinds, seeds=seeds)
        except Exception as e:  # fail-open: memory must not break the host app
            warnings.warn(f"DMem retrieve degraded to empty: {e}", stacklevel=2)
            return []

    # ------------------------------------------------------------------ #
    # Model-switch handoff
    # ------------------------------------------------------------------ #
    def handoff(self, query: str, *, namespace: Optional[str] = None,
                token_budget: Optional[int] = None,
                include_credentials: bool = False) -> Handoff:
        """Build the compact, token-budgeted, priority-ordered handoff for a
        model switch. Flags low-confidence for the hard-fallback path."""
        ns = namespace or self.config.default_namespace
        budget = token_budget or self.config.handoff_token_budget
        results = self.retrieve(query, namespace=ns)
        conflicts = detect_conflicts({r.source_id: r for r in results})
        return build_handoff(
            results, token_budget=budget, conflicts=conflicts,
            low_confidence_threshold=self.config.low_confidence_threshold,
            include_credentials=include_credentials)

    # ------------------------------------------------------------------ #
    # Conversation compaction (long single-model sessions)
    # ------------------------------------------------------------------ #
    def compact(self, transcript: list[dict], *, target_tokens: int = 800,
                namespace: Optional[str] = None) -> str:
        """Compress a long same-model transcript to keep per-turn cost flat.

        Uses the BYO LLM when configured; otherwise an extractive fallback that
        keeps the highest-signal lines within budget. Also opportunistically
        ingests durable facts from the transcript so they persist beyond the
        compacted window."""
        ns = namespace or self.config.default_namespace
        text = "\n".join(f"{m.get('role','user')}: {m.get('content','')}"
                         for m in transcript)
        # persist durable facts before we throw transcript detail away
        for m in transcript:
            if m.get("content"):
                try:
                    self.ingest_message(m["content"], namespace=ns,
                                        role=m.get("role", "user"))
                except Exception:
                    pass

        if self.llm is not None:
            prompt = (f"Summarize this conversation faithfully in about "
                      f"{target_tokens} tokens, preserving decisions, facts, and "
                      f"open threads:\n\n{text}")
            try:
                return self.llm.complete(prompt, max_tokens=target_tokens)
            except Exception:
                pass
        return _extractive_summary(text, target_tokens)

    # ------------------------------------------------------------------ #
    # Admin
    # ------------------------------------------------------------------ #
    def forget(self, namespace: str) -> int:
        """Hard-delete everything for a namespace (right-to-be-forgotten)."""
        return self.store.delete_namespace(namespace)

    def forget_fact(self, fact_id: str) -> bool:
        """Hard-delete a single fact by id, including any closed history for it.

        Unlike contradiction handling (which *closes* a fact's validity window),
        this permanently removes it — for a targeted right-to-be-forgotten
        request that must win over the preserve-history default."""
        return self.store.delete_fact(fact_id)

    def forget_matching(self, namespace: str, *, subject: Optional[str] = None,
                        predicate: Optional[str] = None,
                        object: Optional[str] = None) -> int:
        """Hard-delete all facts (current and closed) matching the given fields.

        At least one of subject/predicate/object must be provided. Use e.g.
        ``forget_matching(ns, subject="Ada Lovelace")`` to erase everything
        recorded about one entity."""
        if not any([subject, predicate, object]):
            raise ValueError("forget_matching needs at least one of "
                             "subject/predicate/object.")
        return self.store.delete_facts(namespace, subject=subject,
                                       predicate=predicate, object=object)

    def close(self) -> None:
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _looks_texty(media_type: Optional[str], filename: Optional[str]) -> bool:
    if media_type and (media_type.startswith("text/")
                       or media_type in ("application/json",
                                         "application/x-ndjson")):
        return True
    if filename:
        low = filename.lower()
        return low.endswith((".md", ".txt", ".csv", ".tsv", ".json", ".log"))
    return False


def _extractive_summary(text: str, target_tokens: int) -> str:
    budget_chars = target_tokens * 4
    if len(text) <= budget_chars:
        return text
    lines = [l for l in text.splitlines() if l.strip()]
    # prioritize lines that look decision/fact bearing
    import re
    key_re = re.compile(r"\b(decide|decided|use|will|must|prefer|name|deadline|"
                        r"because|todo|next|blocker)\b", re.I)
    keyed = sorted(lines, key=lambda l: (bool(key_re.search(l)), len(l)),
                   reverse=True)
    out, used = [], 0
    for l in keyed:
        if used + len(l) > budget_chars:
            continue
        out.append(l)
        used += len(l)
    return "\n".join(out)
