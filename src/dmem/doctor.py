"""`dmem doctor` — preflight validator for the bring-your-own prerequisites.

DMem is bring-your-own-infrastructure and never installs anything. This command
inspects the environment and tells the adopter whether their prerequisites are
in place: a graph DB, a Postgres/pgvector database (and whether pgvector is
actually enabled), and an embedding endpoint. It reports ready / not-ready with
an actionable fix for each gap — it does not provision or install.

Run:  ``dmem-doctor``  (exit 0 when ready, 1 otherwise — usable in CI / an IDE
setup check).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

REQUIRED = "required"
OPTIONAL = "optional"


@dataclass
class Check:
    name: str
    status: str            # "ok" | "fail" | "warn" | "info"
    detail: str = ""
    kind: str = REQUIRED

    @property
    def is_blocking(self) -> bool:
        return self.kind == REQUIRED and self.status == "fail"


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not any(c.is_blocking for c in self.checks)


# --------------------------------------------------------------------------- #
# individual checks (pure — read env, no side effects unless a live probe runs)
# --------------------------------------------------------------------------- #

def _env(env, *keys):
    for k in keys:
        v = env.get(k)
        if v and v.strip():
            return v.strip()
    return None


def _check_embedding(env) -> Check:
    provider = (_env(env, "EMBEDDING_PROVIDER") or "").lower()
    host = _env(env, "EMBEDDING_HOST_URL")
    key = _env(env, "EMBEDDING_API_KEY")
    model = _env(env, "EMBEDDING_MODEL")
    native = provider in ("google", "gemini", "cohere")
    if host:
        return Check("Embedding endpoint", "ok",
                     f"{provider or 'openai-compatible'} @ {host}"
                     + (f" ({model})" if model else ""))
    if native:
        if not key:
            return Check("Embedding endpoint", "fail",
                         f"provider={provider} needs EMBEDDING_API_KEY")
        return Check("Embedding endpoint", "ok", f"{provider} ({model or 'default'})")
    return Check("Embedding endpoint", "fail",
                 "set EMBEDDING_PROVIDER + EMBEDDING_HOST_URL/API_KEY "
                 "(OpenAI-compatible, google, or cohere)")


def _check_graph(env, probe: Optional[Callable] = None) -> Check:
    kind = (_env(env, "GRAPH_DB") or "").lower()
    url = _env(env, "GRAPH_DB_URL")
    if not kind or not url:
        return Check("Graph database", "fail",
                     "set GRAPH_DB (neo4j|falkordb) and GRAPH_DB_URL")
    if kind not in ("neo4j", "falkordb"):
        return Check("Graph database", "fail",
                     f"GRAPH_DB={kind!r} unsupported (use neo4j|falkordb)")
    if probe is None:
        return Check("Graph database", "ok", f"{kind} @ {url} (not probed)")
    ok, detail = probe(kind, url, _env(env, "GRAPH_DB_USER"),
                       _env(env, "GRAPH_DB_PASSWORD"),
                       _env(env, "GRAPH_DB_DATABASE"))
    return Check("Graph database", "ok" if ok else "fail",
                 f"{kind} @ {url} — {detail}")


def _check_pgvector(env, probe: Optional[Callable] = None) -> Check:
    url = _env(env, "PGVECTOR_URL")
    if not url:
        return Check("Postgres + pgvector", "fail",
                     "set PGVECTOR_URL to your existing Postgres database")
    if probe is None:
        return Check("Postgres + pgvector", "ok", f"{_redact(url)} (not probed)")
    ok, detail = probe(url)
    return Check("Postgres + pgvector", "ok" if ok else "fail",
                 f"{_redact(url)} — {detail}")


def _check_optional(env) -> list[Check]:
    out = []
    llm = _env(env, "LLM_HOST_URL")
    out.append(Check("LLM (fact extraction)", "ok" if llm else "info",
                     llm or "not set — offline heuristic extractor (lower recall)",
                     kind=OPTIONAL))
    ocr = _env(env, "OCR_HOST_URL")
    out.append(Check("OCR (binary documents)", "ok" if ocr else "info",
                     ocr or "not set — text/markdown documents only",
                     kind=OPTIONAL))
    return out


# --------------------------------------------------------------------------- #
# live probes (best-effort; require the client extras + reachable servers)
# --------------------------------------------------------------------------- #

def probe_pgvector(url: str) -> tuple[bool, str]:
    try:
        import psycopg
    except ImportError:
        return False, "psycopg not installed (pip install dmem[pgvector])"
    from .stores.pgvector_store import ensure_pgvector
    from .errors import StoreError
    try:
        with psycopg.connect(url, autocommit=True, connect_timeout=5) as conn:
            state = ensure_pgvector(conn)
            return True, f"pgvector {state}"
    except StoreError as e:
        return False, str(e).split("\n")[0]
    except Exception as e:
        return False, f"could not connect: {e}"


def probe_graph(kind, url, user, password, database) -> tuple[bool, str]:
    from .config import GraphConfig, GraphKind
    try:
        from .stores.graph_store import _CypherDriver
        d = _CypherDriver(GraphConfig(kind=GraphKind(kind), url=url, user=user,
                                      password=password, database=database))
        d.run("RETURN 1 AS ok")
        d.close()
        return True, "reachable"
    except Exception as e:
        return False, f"unreachable: {str(e).splitlines()[0]}"


# --------------------------------------------------------------------------- #
# orchestration + CLI
# --------------------------------------------------------------------------- #

def run_checks(env: Optional[dict] = None, *, live: bool = True,
               pg_probe: Optional[Callable] = None,
               graph_probe: Optional[Callable] = None) -> DoctorReport:
    env = dict(os.environ if env is None else env)
    tier = (_env(env, "MEMORY_TIER") or "pro").lower()

    checks: list[Check] = [Check("Tier", "ok" if tier == "pro" else "warn",
                                 tier + ("" if tier == "pro"
                                         else " (DEV/TEST tier — not the "
                                              "supported production setup)"))]

    if tier == "pro":
        pgp = pg_probe if pg_probe is not None else (probe_pgvector if live else None)
        grp = graph_probe if graph_probe is not None else (probe_graph if live else None)
        checks.append(_check_graph(env, probe=grp))
        checks.append(_check_pgvector(env, probe=pgp))
        checks.append(_check_embedding(env))
    else:
        checks.append(Check("Prerequisites", "info",
                            "dev tier — pro prerequisites not enforced",
                            kind=OPTIONAL))
    checks.extend(_check_optional(env))
    return DoctorReport(checks=checks)


def _redact(url: str) -> str:
    # hide credentials in a URL for display
    import re
    return re.sub(r"//[^@/]*@", "//***@", url)


_SYMBOL = {"ok": "✓", "fail": "✗", "warn": "!", "info": "○"}


def format_report(report: DoctorReport) -> str:
    lines = ["DMem preflight (dmem doctor)",
             "─" * 52]
    for c in report.checks:
        lines.append(f"  {_SYMBOL.get(c.status, '?')} {c.name:<22} {c.detail}")
    lines.append("─" * 52)
    if report.ready:
        lines.append("Verdict: READY — all mandatory prerequisites are in place.")
    else:
        gaps = [c for c in report.checks if c.is_blocking]
        lines.append(f"Verdict: NOT READY — {len(gaps)} prerequisite(s) missing:")
        for c in gaps:
            lines.append(f"    • {c.name}: {c.detail}")
        lines.append("\nDMem is bring-your-own — it does not install these. "
                     "See docs/prerequisites.md.")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entry
    import sys
    report = run_checks()
    print(format_report(report))
    sys.exit(0 if report.ready else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
