"""Structure-aware chunking.

Chunk boundaries respect OKF concept sections rather than fixed token windows.
A section that fits within the target size becomes a single chunk. A section
larger than the target is split on paragraph/sentence boundaries (and table rows
are kept whole), never mid-sentence. Small adjacent sections are NOT merged
across concept boundaries — keeping concepts distinct is the point.
"""

from __future__ import annotations

import re
from typing import Optional

from ..types import Chunk, Provenance
from .okf import OKFConcept

_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def _approx_tokens(text: str) -> int:
    # ~4 chars/token heuristic; good enough for budgeting without a tokenizer.
    return max(1, len(text) // 4)


def _split_oversized(text: str, target_tokens: int) -> list[str]:
    """Split a too-large section on paragraph, then sentence, boundaries.

    Table blocks (contiguous ``|...|`` lines) are emitted whole so a table is
    never cut across chunks."""
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush():
        nonlocal buf, buf_tokens
        if buf:
            out.append("\n\n".join(buf).strip())
            buf, buf_tokens = [], 0

    # First separate table blocks from prose so we can keep tables intact.
    blocks = _segment_tables(text)
    for kind, block in blocks:
        if kind == "table":
            # tables emitted whole even if large
            if buf_tokens and buf_tokens + _approx_tokens(block) > target_tokens:
                flush()
            buf.append(block)
            buf_tokens += _approx_tokens(block)
            flush()
            continue
        for para in _PARA_SPLIT_RE.split(block):
            para = para.strip()
            if not para:
                continue
            ptok = _approx_tokens(para)
            if ptok > target_tokens:
                # split the paragraph by sentences
                sent_buf: list[str] = []
                sent_tok = 0
                for sent in _SENT_SPLIT_RE.split(para):
                    stok = _approx_tokens(sent)
                    if sent_tok + stok > target_tokens and sent_buf:
                        out.append(" ".join(sent_buf).strip())
                        sent_buf, sent_tok = [], 0
                    sent_buf.append(sent)
                    sent_tok += stok
                if sent_buf:
                    out.append(" ".join(sent_buf).strip())
                continue
            if buf_tokens + ptok > target_tokens and buf:
                flush()
            buf.append(para)
            buf_tokens += ptok
    flush()
    return [c for c in out if c]


def _segment_tables(text: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    cur_kind: Optional[str] = None
    cur: list[str] = []
    for line in text.splitlines():
        is_table = bool(_TABLE_ROW_RE.match(line))
        kind = "table" if is_table else "prose"
        if cur_kind is None:
            cur_kind = kind
        if kind != cur_kind:
            segments.append((cur_kind, "\n".join(cur)))
            cur, cur_kind = [], kind
        cur.append(line)
    if cur:
        segments.append((cur_kind or "prose", "\n".join(cur)))
    return segments


def chunk_concepts(
    concepts: list[OKFConcept], document_id: str, *, namespace: str = "default",
    target_tokens: int = 320, provenance: Optional[Provenance] = None,
) -> list[Chunk]:
    """Produce chunks from OKF concepts, respecting concept boundaries."""
    chunks: list[Chunk] = []
    ordinal = 0
    for concept in concepts:
        pieces: list[str]
        if _approx_tokens(concept.text) <= target_tokens:
            pieces = [concept.text]
        else:
            pieces = _split_oversized(concept.text, target_tokens)
        for piece in pieces:
            chunks.append(Chunk(
                text=piece, document_id=document_id, concept=concept.concept,
                section=concept.section_path, ordinal=ordinal,
                namespace=namespace, provenance=provenance,
                metadata={**concept.metadata, "concept_ordinal": concept.ordinal},
            ))
            ordinal += 1
    return chunks
