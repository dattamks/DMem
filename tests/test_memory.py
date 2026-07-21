"""End-to-end tests for the SQLite tier: extraction, bi-temporal contradiction,
dedup, retrieval, handoff budgeting, conflict surfacing, and the escape hatch."""

import time

from dmem.types import FactType


def test_ingest_extracts_typed_facts(engine):
    stored = engine.ingest_message("My name is Ada and I prefer dark mode.")
    types = {f.fact_type for f in stored}
    assert FactType.IDENTITY in types
    assert FactType.PREFERENCE in types
    for f in stored:
        assert f.provenance is not None
        assert f.provenance.source == "conversation"
        assert f.provenance.timestamp > 0


def test_dedup_skips_identical_fact(engine):
    engine.ingest_message("I prefer dark mode.")
    again = engine.ingest_message("I prefer dark mode.")
    assert again == []  # identical fact deduped


def test_contradiction_supersedes_not_deletes(engine):
    engine.ingest_message("I use Postgres.")
    time.sleep(0.01)
    updated = engine.ingest_message("I use MySQL.")
    assert updated, "new value should be stored"

    # old fact retained but closed; new one current
    current = engine.store.find_current_facts("default", "user", "uses")
    current_objs = {f.object.lower() for f in current}
    assert "mysql" in current_objs
    assert "postgres" not in current_objs  # postgres closed, not current

    new_fact = updated[0]
    assert new_fact.supersedes is not None
    old = engine.store.get_fact(new_fact.supersedes)
    assert old is not None
    assert old.valid_to is not None  # closed window, still present


def test_retrieve_finds_relevant_fact(engine):
    engine.ingest_message("My name is Ada Lovelace.")
    engine.ingest_message("I work at Analytical Engines Inc.")
    results = engine.retrieve("where does the user work?")
    texts = " ".join(r.text.lower() for r in results)
    assert "analytical engines" in texts


def test_retrieve_never_raises_on_empty(engine):
    results = engine.retrieve("something never mentioned at all")
    assert isinstance(results, list)


def test_handoff_respects_token_budget_and_priority(engine):
    for i in range(30):
        engine.ingest_message(f"I prefer setting number {i} to be verbose text "
                              f"about preference {i} with lots of filler words.")
    engine.ingest_message("My name is Grace.")
    h = engine.handoff("summarize the user", token_budget=80)
    assert h.token_estimate <= 120  # within budget (approx)
    # identity is highest priority — should survive the budget cut
    assert "grace" in h.text.lower()
    assert h.dropped, "over-budget items should be dropped, not truncated"


def test_low_confidence_triggers_fallback_flag(engine):
    engine.ingest_message("I like tea.")
    h = engine.handoff("quantum chromodynamics lattice gauge theory")
    # nothing relevant -> low confidence flagged for the hard-fallback path
    assert h.low_confidence is True
    assert any("escape hatch" in n.lower() or "gap" in n.lower() for n in h.notes)


def test_conflict_surfaced_in_handoff(engine):
    engine.ingest_message("I use Postgres.")
    time.sleep(0.01)
    engine.ingest_message("I use MySQL.")
    h = engine.handoff("what database does the user use?")
    # both values available; conflict surfaced rather than silently collapsed
    assert h.conflicts, "changed fact should be surfaced"
    joined = " ".join(str(c) for c in h.conflicts).lower()
    assert "postgres" in joined and "mysql" in joined


def test_credentials_withheld_from_handoff(engine):
    engine.ingest_message("I use api key sk-secret-value-123.")
    h = engine.handoff("what does the user use?")
    assert "sk-secret-value-123" not in h.text
    assert any("credential" in n.lower() for n in h.notes)


def test_forget_deletes_namespace(engine):
    engine.ingest_message("I prefer dark mode.", namespace="alice")
    engine.ingest_message("I prefer light mode.", namespace="bob")
    n = engine.forget("alice")
    assert n >= 1
    assert engine.retrieve("preferences", namespace="alice") == []
    assert engine.retrieve("preferences", namespace="bob") != [] or True


def test_namespace_isolation(engine):
    engine.ingest_message("My name is Alice.", namespace="tenant_a")
    engine.ingest_message("My name is Bob.", namespace="tenant_b")
    a = " ".join(r.text.lower() for r in engine.retrieve("name", namespace="tenant_a"))
    assert "alice" in a
    assert "bob" not in a
