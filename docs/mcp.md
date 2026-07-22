# DMem as an MCP server

Expose DMem's memory to any MCP client (Claude Desktop, Cursor, …) with no
per-client code.

## Install & run

```bash
pip install "dmem[mcp]"      # add [http] for remote embeddings, etc.
dmem-mcp                     # stdio transport; configured via env vars
```

## Tools exposed

| Tool | Purpose |
|---|---|
| `dmem_recall` | Retrieve relevant facts + document context (the **escape hatch** — call it mid-conversation when context is missing). |
| `dmem_remember` | Persist durable facts from a message. |
| `dmem_handoff` | Get a compact, token-budgeted model-switch handoff. |
| `dmem_ingest_document` | Index a text/markdown document. |
| `dmem_forget` | Permanently delete a namespace. |

## Claude Desktop config

```json
{
  "mcpServers": {
    "dmem": {
      "command": "dmem-mcp",
      "env": {
        "MEMORY_TIER": "sqlite",
        "SQLITE_PATH": "/absolute/path/to/dmem.db",
        "EMBEDDING_HOST_URL": "https://api.openai.com/v1",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_API_KEY": "sk-..."
      }
    }
  }
}
```

Omit the embedding vars to run fully offline (dev-quality hashing embedder).
See [`../.env.example`](../.env.example) for every option and the higher tiers.
