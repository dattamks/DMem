"""Live MCP round-trip: a REAL client session talking to the DMem MCP server.

Uses the MCP SDK's in-memory client<->server transport (the same protocol path a
VS Code / Claude / Cursor client uses, minus the OS pipe) to prove the handshake,
tool listing, and tool calls actually work end to end — not just the dispatch
logic. Skips if the `mcp` SDK isn't installed.

    pip install -e ".[mcp,dev]"
    pytest tests/integration/test_mcp_live.py -q
"""

import json
import warnings

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session

from dmem import Config, DMemEngine, Tier
from dmem.config import EmbeddingConfig
from dmem.adapters.mcp_server import build_server


def _engine(tmp_path):
    cfg = Config(tier=Tier.SQLITE, sqlite_path=str(tmp_path / "mcp.db"),
                 embedding=EmbeddingConfig())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DMemEngine(cfg)


def _text(result):
    # tool results carry a list of content blocks; pull the text payload
    return "".join(getattr(c, "text", "") for c in result.content)


@pytest.mark.asyncio
async def test_mcp_handshake_and_tool_flow(tmp_path):
    engine = _engine(tmp_path)
    server = build_server(engine)
    try:
        async with create_connected_server_and_client_session(server) as client:
            # 1) handshake
            await client.initialize()

            # 2) the server advertises DMem's tools
            tools = await client.list_tools()
            names = {t.name for t in tools.tools}
            assert {"dmem_recall", "dmem_remember", "dmem_handoff",
                    "dmem_ingest_document", "dmem_forget"} <= names

            # 3) remember -> real tool call over the protocol
            r = await client.call_tool(
                "dmem_remember",
                {"text": "My name is Ada and I work at Analytical Engines.",
                 "namespace": "mcp1"})
            stored = json.loads(_text(r))
            assert stored["count"] >= 1

            # 4) recall finds it
            r2 = await client.call_tool(
                "dmem_recall",
                {"query": "where does the user work?", "namespace": "mcp1"})
            recalled = json.loads(_text(r2))
            assert recalled["count"] >= 1
            assert any("analytical engines" in item["text"].lower()
                       for item in recalled["results"])

            # 5) handoff returns a compact, budgeted context blob
            r3 = await client.call_tool(
                "dmem_handoff",
                {"query": "who is the user?", "namespace": "mcp1"})
            handoff = json.loads(_text(r3))
            assert "ada" in handoff["text"].lower()
            assert handoff["token_estimate"] > 0
    finally:
        engine.close()


@pytest.mark.asyncio
async def test_mcp_document_ingest_and_forget(tmp_path):
    engine = _engine(tmp_path)
    server = build_server(engine)
    try:
        async with create_connected_server_and_client_session(server) as client:
            await client.initialize()
            r = await client.call_tool(
                "dmem_ingest_document",
                {"text": "# Runbook\n\nProd runs Postgres 16 in AWS us-east-1.",
                 "document_id": "rb", "namespace": "mcp2"})
            assert json.loads(_text(r))["indexed_chunks"] >= 1

            r2 = await client.call_tool(
                "dmem_recall",
                {"query": "where is prod hosted?", "namespace": "mcp2",
                 "kinds": ["chunk"]})
            assert "us-east-1" in _text(r2).lower()

            r3 = await client.call_tool("dmem_forget", {"namespace": "mcp2"})
            assert json.loads(_text(r3))["deleted"] >= 1
    finally:
        engine.close()
