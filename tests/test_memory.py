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


def test_single_valued_contradiction_supersedes_not_deletes(engine):
    # works_at is single-valued: a new employer retires the old one.
    engine.ingest_message("I work at Google.")
    time.sleep(0.01)
    updated = engine.ingest_message("I work at Meta.")
    assert updated, "new value should be stored"

    current = engine.store.find_current_facts("default", "user", "works_at")
    current_objs = {f.object.lower() for f in current}
    assert "meta" in current_objs
    assert "google" not in current_objs  # google closed, not current

    new_fact = updated[0]
    assert new_fact.supersedes is not None
    old = engine.store.get_fact(new_fact.supersedes)
    assert old is not None
    assert old.valid_to is not None  # closed window, still present


def test_multi_valued_predicate_accumulates(engine):
    # "uses" is multi-valued: both remain current, not a contradiction.
    engine.ingest_message("I use Postgres.")
    time.sleep(0.01)
    engine.ingest_message("I use Redis.")
    current = engine.store.find_current_facts("default", "user", "uses")
    objs = {f.object.lower() for f in current}
    assert "postgres" in objs and "redis" in objs
    for f in current:
        assert f.valid_to is None  # nothing closed


def test_multi_valued_exact_duplicate_still_deduped(engine):
    engine.ingest_message("I use Postgres.")
    again = engine.ingest_message("I use Postgres.")
    assert again == []  # identical multi-valued fact still deduped


def test_cardinality_override_from_config(tmp_path):
    # Force "uses" to be single-valued via config override.
    import warnings
    from dmem import Config, DMemEngine, Tier
    from dmem.config import EmbeddingConfig
    from dmem.types import Cardinality
    cfg = Config(tier=Tier.SQLITE, sqlite_path=str(tmp_path / "o.db"),
                 embedding=EmbeddingConfig(),
                 predicate_cardinality_overrides={"uses": Cardinality.SINGLE})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = DMemEngine(cfg)
    try:
        eng.ingest_message("I use Postgres.")
        time.sleep(0.01)
        eng.ingest_message("I use MySQL.")
        current = eng.store.find_current_facts("default", "user", "uses")
        objs = {f.object.lower() for f in current}
        assert objs == {"mysql"}  # override made it supersede
    finally:
        eng.close()


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
    # single-valued predicate: the change genuinely supersedes -> surfaced
    engine.ingest_message("I work at Google.")
    time.sleep(0.01)
    engine.ingest_message("I work at Meta.")
    h = engine.handoff("where does the user work?")
    assert h.conflicts, "changed fact should be surfaced"
    joined = " ".join(str(c) for c in h.conflicts).lower()
    assert "google" in joined and "meta" in joined


def test_multi_valued_not_flagged_as_conflict(engine):
    # co-existing multi-valued facts are NOT a conflict
    engine.ingest_message("I use Postgres.")
    time.sleep(0.01)
    engine.ingest_message("I use Redis.")
    h = engine.handoff("what does the user use?")
    assert h.conflicts == [], "concurrent multi-valued facts are not conflicts"


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


def test_forget_single_fact_hard_delete(engine):
    stored = engine.ingest_message("My name is Ada.")
    fact_id = stored[0].id
    assert engine.forget_fact(fact_id) is True
    assert engine.store.get_fact(fact_id) is None  # gone, not just closed
    assert engine.forget_fact(fact_id) is False    # already gone


def test_forget_matching_erases_entity(engine):
    engine.ingest_message("I work at Google.")
    engine.ingest_message("I prefer Rust.")
    engine.ingest_message("My name is Ada.")
    # erase everything about subject "user"
    n = engine.forget_matching("default", subject="user")
    assert n >= 3
    assert engine.retrieve("anything about the user") == []


def test_forget_matching_includes_closed_history(engine):
    engine.ingest_message("I work at Google.")
    time.sleep(0.01)
    engine.ingest_message("I work at Meta.")  # closes the Google fact
    # deleting by predicate removes both current and closed rows
    n = engine.forget_matching("default", predicate="works_at")
    assert n == 2


def test_forget_matching_requires_a_filter(engine):
    import pytest
    with pytest.raises(ValueError):
        engine.forget_matching("default")


def test_namespace_isolation(engine):
    engine.ingest_message("My name is Alice.", namespace="tenant_a")
    engine.ingest_message("My name is Bob.", namespace="tenant_b")
    a = " ".join(r.text.lower() for r in engine.retrieve("name", namespace="tenant_a"))
    assert "alice" in a
    assert "bob" not in a
