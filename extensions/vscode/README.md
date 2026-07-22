# DMem VS Code extension (scaffold)

A thin IDE surface over the DMem **MCP server** — it contains no memory logic of
its own. It launches `dmem-mcp` as a child process and lets the editor's AI
features recall/remember through DMem's tools.

## Status

Scaffold / work-in-progress. The design is intentionally minimal: the extension
is an MCP client that spawns `dmem-mcp` and forwards the tool set
(`dmem_recall`, `dmem_remember`, `dmem_handoff`, `dmem_ingest_document`,
`dmem_forget`). All configuration is via the same environment variables the SDK
and MCP server use (see [`../../.env.example`](../../.env.example)) so behavior
is identical across surfaces.

## Planned commands

| Command | Action |
|---|---|
| `DMem: Recall…` | Run `dmem_recall` on a prompt; show results. |
| `DMem: Remember Selection` | Persist the selected text as facts. |
| `DMem: Ingest File` | Send the active file through `dmem_ingest_document`. |
| `DMem: Handoff` | Produce a compact handoff for the current context. |
| `DMem: Forget Namespace…` | Delete a namespace. |

## Prerequisite

```bash
pip install "dmem[mcp]"   # provides the dmem-mcp server the extension spawns
```

See [`package.json`](package.json) for the (stub) contribution points.
