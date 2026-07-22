"""Retrieval-quality smoke harness.

A lightweight, dependency-free evaluation that gives adopters a day-one signal on
retrieval quality for *their* configuration — without the full LongMemEval /
LoCoMo setup (which remains the deferred, formal benchmarking track).

Two things it measures:
- **Retrieval metrics** (hit@k, recall@k, MRR) over a small labelled dataset.
- **Behavioral checks** (contradiction supersede, multi-valued accumulation,
  conflict surfacing) that assert the memory semantics hold end-to-end.

Run the built-in smoke suite against your env config:

    dmem-eval               # human-readable report
    dmem-eval --json        # machine-readable

Or programmatically:

    from dmem.eval import run_smoke
    report = run_smoke()          # uses Config.from_env()
    print(report.hit_at_k)

The offline hashing embedder gives a *floor*; pointing EMBEDDING_HOST_URL at a
real model should raise the scores — that delta is the point.
"""

from .dataset import SMOKE_SCENARIOS, Scenario, Query
from .harness import EvalReport, ScenarioResult, run, run_smoke

__all__ = [
    "SMOKE_SCENARIOS", "Scenario", "Query",
    "EvalReport", "ScenarioResult", "run", "run_smoke",
]
