"""Escape-hatch tool dispatch (the retrieval-as-a-tool surface)."""

from dmem.tools.escape_hatch import TOOL_SPECS, dispatch_tool


def test_tool_specs_present():
    names = {t["name"] for t in TOOL_SPECS}
    assert {"dmem_recall", "dmem_remember", "dmem_handoff"} <= names


def test_recall_dispatch(engine):
    engine.ingest_message("My name is Ada.")
    out = dispatch_tool(engine, "dmem_recall", {"query": "user's name"})
    assert out["count"] >= 1
    assert out["empty"] is False


def test_recall_empty_advises_not_to_guess(engine):
    out = dispatch_tool(engine, "dmem_recall", {"query": "unknown topic xyzzy"})
    assert out["empty"] is True
    assert "ask the user" in out["note"].lower()


def test_remember_dispatch(engine):
    out = dispatch_tool(engine, "dmem_remember",
                        {"text": "I prefer tabs over spaces."})
    assert out["count"] >= 1


def test_handoff_dispatch(engine):
    engine.ingest_message("My name is Ada and I use Rust.")
    out = dispatch_tool(engine, "dmem_handoff", {"query": "user profile"})
    assert "text" in out
    assert "token_estimate" in out
