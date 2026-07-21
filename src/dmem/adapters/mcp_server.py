"""MCP server adapter.

Exposes the DMem engine as MCP tools (``dmem_recall``, ``dmem_remember``,
``dmem_handoff``, ``dmem_ingest_document``, ``dmem_forget``) so any MCP client —
Claude Desktop, Cursor, and others — gets memory with no per-client code.

Run:  ``dmem-mcp``  (stdio transport), configured entirely via env vars.

Requires the ``mcp`` extra: ``pip install dmem[mcp]``. Kept behind a lazy import
so the core package doesn't depend on the MCP SDK.
"""

from __future__ import annotations

import json
from typing import Any

from ..engine import DMemEngine
from ..tools.escape_hatch import TOOL_SPECS, dispatch_tool


def build_server(engine: DMemEngine):
    """Construct an MCP ``Server`` wired to the engine. Returns the server."""
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as e:  # pragma: no cover - depends on extra
        raise ImportError(
            "The MCP server adapter needs the 'mcp' extra: pip install dmem[mcp]"
        ) from e

    server = Server("dmem")

    extra_tools = [
        {
            "name": "dmem_ingest_document",
            "description": ("Ingest a text/markdown document into memory "
                            "(OCR-backed binary ingest is available via the SDK)."),
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "document_id": {"type": "string"},
                    "namespace": {"type": "string"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "dmem_forget",
            "description": "Permanently delete all memory for a namespace.",
            "input_schema": {
                "type": "object",
                "properties": {"namespace": {"type": "string"}},
                "required": ["namespace"],
            },
        },
    ]
    all_specs = TOOL_SPECS + extra_tools

    @server.list_tools()
    async def list_tools() -> list[Any]:
        return [Tool(name=s["name"], description=s["description"],
                     inputSchema=s["input_schema"]) for s in all_specs]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[Any]:
        result = _dispatch(engine, name, arguments)
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    return server


def _dispatch(engine: DMemEngine, name: str, arguments: dict) -> dict:
    if name == "dmem_ingest_document":
        chunks = engine.ingest_document(
            arguments["text"], document_id=arguments.get("document_id"),
            namespace=arguments.get("namespace"))
        return {"indexed_chunks": len(chunks)}
    if name == "dmem_forget":
        n = engine.forget(arguments["namespace"])
        return {"deleted": n}
    return dispatch_tool(engine, name, arguments)


def main() -> None:  # pragma: no cover - entry point
    """Console-script entry point: run the stdio MCP server."""
    import asyncio
    import sys

    from ..errors import ConfigError

    from mcp.server.stdio import stdio_server

    try:
        engine = DMemEngine()
    except ConfigError as e:
        # Missing bring-your-own prerequisites: print a clear preflight to stderr
        # (stdout is the MCP stream) instead of a stack trace, then exit.
        from ..doctor import format_report, run_checks
        print("DMem MCP server cannot start — prerequisites missing.\n",
              file=sys.stderr)
        print(format_report(run_checks(live=False)), file=sys.stderr)
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)

    async def _run():
        server = build_server(engine)
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    try:
        asyncio.run(_run())
    finally:
        engine.close()


if __name__ == "__main__":  # pragma: no cover
    main()
