"""Tool definitions exposed to LLMs (escape hatch, ingest, handoff)."""

from .escape_hatch import TOOL_SPECS, dispatch_tool

__all__ = ["TOOL_SPECS", "dispatch_tool"]
