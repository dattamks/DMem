# DMem image — runs the MCP server (default), the OpenAI-compatible proxy, or
# the eval CLI. The Python SDK is a library; this image is for the *server*
# surfaces. See docs/docker.md.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy metadata first for better layer caching, then the source.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Install with the server-oriented extras. Add ,rerank for cross-encoder
# reranking (pulls torch — much larger image) if you need it.
RUN pip install ".[http,mcp,server,neo4j,falkordb,pgvector]"

# SQLite tier writes here by default; mount a volume to persist it.
ENV SQLITE_PATH=/data/dmem.db
VOLUME ["/data"]

# Non-root for safety.
RUN useradd -m dmem && mkdir -p /data && chown dmem /data
USER dmem

EXPOSE 8000

# Default: the MCP server over stdio. Override the command to run the proxy:
#   docker run -p 8000:8000 dmem uvicorn dmem.adapters.proxy:app --host 0.0.0.0 --port 8000
# or the eval CLI:  docker run dmem dmem-eval
CMD ["dmem-mcp"]
