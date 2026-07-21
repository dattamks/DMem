"""Labelled scenarios for the smoke harness.

A scenario seeds an isolated namespace (messages + documents), then runs:
- retrieval `Query` items scored for hit@k / recall@k / MRR, and
- optional `checks` (callables) that assert memory *behavior* end-to-end.

Relevance is judged by case-insensitive substring: a retrieved item is relevant
to a query if it contains any of the query's `relevant` phrases. This keeps the
dataset embedder-agnostic (no gold ids to maintain).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Query:
    text: str
    relevant: list[str]                       # phrases that SHOULD be retrieved
    kinds: tuple[str, ...] = ("fact", "chunk")
    must_not: list[str] = field(default_factory=list)  # phrases that must NOT appear


@dataclass
class Doc:
    text: str
    document_id: str


@dataclass
class Scenario:
    name: str
    messages: list[str] = field(default_factory=list)
    documents: list[Doc] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)
    # behavioral checks: (engine, namespace) -> (label, passed, detail)
    checks: list[Callable] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# behavioral check helpers
# --------------------------------------------------------------------------- #

def _current_objects(engine, ns, subject, predicate) -> set[str]:
    return {f.object.lower()
            for f in engine.store.find_current_facts(ns, subject, predicate)}


def check_contradiction_supersedes(engine, ns):
    objs = _current_objects(engine, ns, "user", "works_at")
    passed = objs == {"meta"}
    return ("single-valued supersede (works_at)", passed,
            f"current works_at={sorted(objs)} (expected just 'meta')")


def check_multi_valued_accumulates(engine, ns):
    objs = _current_objects(engine, ns, "user", "uses")
    passed = {"postgres", "redis"} <= objs
    return ("multi-valued accumulate (uses)", passed,
            f"current uses={sorted(objs)} (expected postgres & redis)")


def check_conflict_surfaced(engine, ns):
    h = engine.handoff("where does the user work?", namespace=ns)
    joined = " ".join(str(c) for c in h.conflicts).lower()
    passed = bool(h.conflicts) and "google" in joined and "meta" in joined
    return ("conflict surfaced in handoff", passed,
            f"conflicts={len(h.conflicts)}")


def check_credential_not_leaked(engine, ns):
    h = engine.handoff("what does the user use?", namespace=ns)
    leaked = "sk-secret-eval-42" in h.text
    stored_raw = any("sk-secret-eval-42" in f.object for f in engine.store.iter_facts()
                     if f.namespace == ns)
    return ("credential redacted, not leaked", (not leaked) and (not stored_raw),
            f"in_handoff={leaked} stored_raw={stored_raw}")


# --------------------------------------------------------------------------- #
# built-in smoke suite
# --------------------------------------------------------------------------- #

_RUNBOOK = """# Deployment Runbook

## Hosting

Production runs Postgres 16 in AWS us-east-1. Staging is in us-west-2.

## On-call

The primary on-call rotation is weekly; escalate to the platform team after
15 minutes with no ack.

## Backups

Nightly snapshots are retained for 30 days in an encrypted S3 bucket.
"""

SMOKE_SCENARIOS: list[Scenario] = [
    Scenario(
        name="identity_and_preferences",
        messages=[
            "My name is Ada Lovelace.",
            "I work at Analytical Engines Inc.",
            "I prefer Rust and I use Postgres.",
        ],
        queries=[
            Query("what is the user's name?", ["ada lovelace"], kinds=("fact",)),
            Query("where does the user work?", ["analytical engines"], kinds=("fact",)),
            Query("what programming language does the user prefer?", ["rust"],
                  kinds=("fact",)),
        ],
    ),
    Scenario(
        name="document_rag",
        documents=[Doc(_RUNBOOK, "runbook")],
        queries=[
            Query("where is production hosted?", ["us-east-1"], kinds=("chunk",)),
            Query("how long are backups kept?", ["30 days"], kinds=("chunk",)),
            Query("what is the on-call escalation policy?",
                  ["escalate", "platform team"], kinds=("chunk",)),
        ],
    ),
    Scenario(
        name="contradiction_supersede",
        messages=["I work at Google.", "I work at Meta."],
        queries=[
            Query("where does the user work now?", ["meta"], kinds=("fact",),
                  must_not=[]),
        ],
        checks=[check_contradiction_supersedes, check_conflict_surfaced],
    ),
    Scenario(
        name="multi_valued_tools",
        messages=["I use Postgres.", "I use Redis.", "I use Kafka."],
        queries=[
            Query("what data stores does the user use?", ["postgres"],
                  kinds=("fact",)),
        ],
        checks=[check_multi_valued_accumulates],
    ),
    Scenario(
        name="credential_safety",
        messages=["I use api key sk-secret-eval-42 for the billing service."],
        queries=[],
        checks=[check_credential_not_leaked],
    ),
]
