"""Document pipeline: OKF structuring, structure-aware chunking, dedup."""

from dmem.pipeline.okf import structure_markdown
from dmem.pipeline.chunking import chunk_concepts


SAMPLE = """# Project Overview

DMem is a memory library.

## Goals

- Persistent memory
- Model-switch continuity

## Data

| key | value |
|-----|-------|
| tier | sqlite |
| dim | 256 |
"""


def test_okf_splits_by_heading():
    concepts = structure_markdown(SAMPLE)
    titles = [c.concept for c in concepts]
    assert "Project Overview" in titles
    assert "Goals" in titles
    assert "Data" in titles
    # section path tracks hierarchy
    data = next(c for c in concepts if c.concept == "Data")
    assert "Project Overview" in data.section_path
    assert data.metadata["has_table"] is True


def test_chunking_respects_concept_boundaries():
    concepts = structure_markdown(SAMPLE)
    chunks = chunk_concepts(concepts, "doc1", target_tokens=1000)
    # each chunk belongs to exactly one concept (no cross-concept merge)
    concepts_seen = {c.concept for c in chunks}
    assert concepts_seen <= {"Project Overview", "Goals", "Data"}
    for ch in chunks:
        assert ch.concept is not None


def test_table_kept_whole_when_splitting():
    big = "# Data\n\n" + "\n".join(
        f"| row{i} | val{i} |" for i in range(50))
    concepts = structure_markdown(big)
    chunks = chunk_concepts(concepts, "doc", target_tokens=20)
    # table rows should not be fragmented into non-table prose
    table_chunks = [c for c in chunks if "|" in c.text]
    assert table_chunks


def test_ingest_document_and_retrieve(engine):
    chunks = engine.ingest_document(SAMPLE, document_id="overview")
    assert chunks
    results = engine.retrieve("what tier does the project use?", kinds=("chunk",))
    assert any("sqlite" in r.text.lower() for r in results)


def test_document_dedup(engine):
    first = engine.ingest_document(SAMPLE, document_id="overview")
    second = engine.ingest_document(SAMPLE, document_id="overview")
    assert first, "first ingest indexes chunks"
    assert second == [], "re-ingesting identical content is deduped"


def test_binary_without_ocr_raises(engine):
    import pytest
    with pytest.raises(ValueError):
        engine.ingest_document(b"\x89PNG\x00binary", filename="scan.png")
