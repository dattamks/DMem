"""Escape-hatch tool: retrieval exposed as a callable tool.

The model can call ``dmem_recall`` mid-conversation when it senses a context
gap, rather than the handoff being a one-shot, unrecoverable bet. These specs
are provider-neutral JSON-Schema tool definitions; the MCP adapter and any
framework adapter translate them to their native tool format.
"""

from __future__ import annotations

from typing import Any

from ..engine import DMemEngine

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "dmem_recall",
        "description": (
            "Retrieve relevant remembered facts and document context for the "
            "current query. Call this whenever you are missing context, the user "
            "references something from earlier, or you just switched models."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "What you need context about."},
                "namespace": {"type": "string",
                              "description": "User/tenant memory namespace."},
                "kinds": {"type": "array", "items": {"type": "string",
                          "enum": ["fact", "chunk"]},
                          "description": "Limit to facts, document chunks, or both."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "dmem_remember",
        "description": ("Persist a durable fact from the conversation so it is "
                        "available across sessions and model switches."),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string",
                         "description": "The message/statement to extract facts from."},
                "namespace": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "dmem_handoff",
        "description": ("Get a compact, token-budgeted context handoff for "
                        "continuing after a model switch."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "namespace": {"type": "string"},
            },
            "required": ["query"],
        },
    },
]


def dispatch_tool(engine: DMemEngine, name: str, arguments: dict) -> dict:
    """Execute a tool call by name against an engine. Returns a JSON-able dict."""
    ns = arguments.get("namespace")
    if name == "dmem_recall":
        kinds = tuple(arguments.get("kinds") or ("fact", "chunk"))
        results = engine.retrieve(arguments["query"], namespace=ns, kinds=kinds)
        return {
            "results": [r.to_dict() for r in results],
            "count": len(results),
            "empty": len(results) == 0,
            "note": ("No relevant memory found — ask the user rather than "
                     "guessing." if not results else ""),
        }
    if name == "dmem_remember":
        stored = engine.ingest_message(arguments["text"], namespace=ns)
        return {"stored": [f.to_dict() for f in stored], "count": len(stored)}
    if name == "dmem_handoff":
        h = engine.handoff(arguments["query"], namespace=ns)
        return h.to_dict()
    raise ValueError(f"Unknown tool: {name}")
