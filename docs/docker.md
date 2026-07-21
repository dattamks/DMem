# Running DMem with Docker

DMem ships three things; only two of them are servers you run in a container.

| Surface | What it is | How you use it |
|---|---|---|
| **SDK** | a Python *library* (`import dmem`) | `pip install dmem` inside your app's image — nothing to run standalone |
| **MCP server** | a stdio process (`dmem-mcp`) | an MCP client launches it as `docker run -i … dmem-mcp` |
| **Proxy** | an HTTP server (`uvicorn dmem.adapters.proxy:app`) | run it and point your app's base URL at it |

## Quick start (proxy + FalkorDB)

```bash
cp .env.example .env      # set EMBEDDING_HOST_URL, PROXY_UPSTREAM_URL, keys
docker compose up --build
```

This starts **FalkorDB** (graph backend, consolidated tier) and the **proxy** on
`http://localhost:8000`. Point any OpenAI client at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
client.chat.completions.create(model="gpt-4o", user="alice",
    messages=[{"role": "user", "content": "remind me where I work"}])
```

## Build the image directly

```bash
docker build -t dmem .
```

Run whichever surface you want (the image has all three entrypoints on PATH):

```bash
# Proxy (HTTP)
docker run --rm -p 8000:8000 \
  -e MEMORY_TIER=sqlite -e SQLITE_PATH=/data/dmem.db -v dmem-data:/data \
  -e PROXY_UPSTREAM_URL=https://api.openai.com/v1 -e PROXY_UPSTREAM_API_KEY=sk-... \
  dmem uvicorn dmem.adapters.proxy:app --host 0.0.0.0 --port 8000

# Eval CLI
docker run --rm dmem dmem-eval

# MCP server (stdio) — normally launched by the client, see below
docker run --rm -i dmem dmem-mcp
```

## MCP server via Docker (stdio)

The MCP server talks over stdin/stdout, so an MCP client spawns it as a
subprocess. Configure the client to run the container with `-i`:

```json
{
  "mcpServers": {
    "dmem": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "dmem-data:/data",
        "-e", "MEMORY_TIER=sqlite",
        "-e", "SQLITE_PATH=/data/dmem.db",
        "-e", "EMBEDDING_HOST_URL=https://api.openai.com/v1",
        "-e", "EMBEDDING_MODEL=text-embedding-3-small",
        "-e", "EMBEDDING_API_KEY=sk-...",
        "dmem", "dmem-mcp"
      ]
    }
  }
}
```

The `-v dmem-data:/data` volume persists memory across container restarts.

## Tiers with Compose

- **Default (consolidated):** FalkorDB is started and the proxy points at it via
  `GRAPH_DB=falkordb`, `GRAPH_DB_URL=redis://falkordb:6379`.
- **SQLite tier:** set `MEMORY_TIER=sqlite` on the service and drop the FalkorDB
  dependency — good for a single-container deploy.
- **Pro tier:** `docker compose --profile pro up` also starts Neo4j and
  Postgres+pgvector; set `MEMORY_TIER=pro`, `PGVECTOR_URL=postgresql://postgres:
  password@postgres:5432/dmem`, `GRAPH_DB=neo4j`,
  `GRAPH_DB_URL=bolt://neo4j:7687`, `GRAPH_DB_USER=neo4j`,
  `GRAPH_DB_PASSWORD=password`.

## Running the graph integration tests against Compose

This is the environment to validate the FalkorDB/Neo4j dialects that CI-gated
tests cover (see [testing.md](testing.md)):

```bash
docker compose up -d falkordb
pip install -e ".[falkordb,dev]"
export DMEM_TEST_GRAPH_KIND=falkordb DMEM_TEST_GRAPH_URL=redis://localhost:6379
pytest tests/integration -q
```

## Notes

- The default image excludes the `rerank` extra (it pulls torch). Add `,rerank`
  to the `pip install` line in the Dockerfile if you want cross-encoder
  reranking in-container.
- The container runs as a non-root `dmem` user; the SQLite DB lives in the
  `/data` volume.
