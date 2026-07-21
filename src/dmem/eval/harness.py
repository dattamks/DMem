"""Runner + metrics for the smoke harness.

Metrics (per query, then averaged):
- hit@k    : 1.0 if any relevant phrase appears in the top-k results
- recall@k : fraction of the query's relevant phrases found in top-k
- MRR      : reciprocal rank of the first relevant result (0 if none)
- clean    : 1.0 if no `must_not` phrase leaked into results

Behavioral checks are pass/fail and reported separately. The overall exit signal
combines: retrieval below a floor OR any behavioral failure => non-zero.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..config import Config
from ..engine import DMemEngine
from .dataset import SMOKE_SCENARIOS, Doc, Query, Scenario


@dataclass
class QueryResult:
    query: str
    hit: float
    recall: float
    mrr: float
    clean: float
    retrieved: list[str]


@dataclass
class CheckResult:
    label: str
    passed: bool
    detail: str


@dataclass
class ScenarioResult:
    name: str
    queries: list[QueryResult] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)


@dataclass
class EvalReport:
    scenarios: list[ScenarioResult]
    k: int
    hit_at_k: float
    recall_at_k: float
    mrr: float
    checks_passed: int
    checks_total: int
    n_queries: int
    duration_s: float
    embedder: str

    @property
    def all_checks_passed(self) -> bool:
        return self.checks_passed == self.checks_total

    def passed(self, hit_floor: float = 0.5) -> bool:
        """Overall pass: retrieval clears the floor AND all behavioral checks."""
        retrieval_ok = self.n_queries == 0 or self.hit_at_k >= hit_floor
        return retrieval_ok and self.all_checks_passed

    def to_dict(self) -> dict:
        return {
            "k": self.k, "hit_at_k": self.hit_at_k,
            "recall_at_k": self.recall_at_k, "mrr": self.mrr,
            "checks_passed": self.checks_passed, "checks_total": self.checks_total,
            "n_queries": self.n_queries, "duration_s": self.duration_s,
            "embedder": self.embedder,
            "scenarios": [
                {
                    "name": s.name,
                    "queries": [vars(q) for q in s.queries],
                    "checks": [vars(c) for c in s.checks],
                } for s in self.scenarios
            ],
        }


def _score_query(retrieved_texts: list[str], q: Query) -> QueryResult:
    lowered = [t.lower() for t in retrieved_texts]
    phrases = [p.lower() for p in q.relevant]

    found = {p for p in phrases if any(p in t for t in lowered)}
    recall = (len(found) / len(phrases)) if phrases else 1.0
    hit = 1.0 if (found or not phrases) else 0.0

    mrr = 0.0
    for rank, t in enumerate(lowered, start=1):
        if any(p in t for p in phrases):
            mrr = 1.0 / rank
            break

    leaked = any(m.lower() in t for m in q.must_not for t in lowered)
    clean = 0.0 if leaked else 1.0
    return QueryResult(query=q.text, hit=hit, recall=recall, mrr=mrr,
                       clean=clean, retrieved=retrieved_texts)


def run(engine: DMemEngine, scenarios: Optional[list[Scenario]] = None, *,
        k: Optional[int] = None, namespace_prefix: str = "eval") -> EvalReport:
    """Run scenarios against an engine and return a metrics report."""
    scenarios = scenarios if scenarios is not None else SMOKE_SCENARIOS
    k = k or engine.config.rerank_top_k
    start = time.time()

    results: list[ScenarioResult] = []
    all_q: list[QueryResult] = []
    checks_passed = checks_total = 0

    for sc in scenarios:
        ns = f"{namespace_prefix}_{sc.name}"
        engine.forget(ns)  # isolate + make re-runs idempotent

        for msg in sc.messages:
            engine.ingest_message(msg, namespace=ns)
        for doc in sc.documents:
            engine.ingest_document(doc.text, document_id=doc.document_id,
                                   namespace=ns)

        sr = ScenarioResult(name=sc.name)
        for q in sc.queries:
            hits = engine.retrieve(q.text, namespace=ns, kinds=q.kinds)
            qr = _score_query([h.text for h in hits[:k]], q)
            sr.queries.append(qr)
            all_q.append(qr)

        for check in sc.checks:
            label, passed, detail = check(engine, ns)
            sr.checks.append(CheckResult(label=label, passed=passed, detail=detail))
            checks_total += 1
            checks_passed += 1 if passed else 0

        results.append(sr)
        engine.forget(ns)  # tidy up

    n = len(all_q)
    hit = sum(q.hit for q in all_q) / n if n else 0.0
    recall = sum(q.recall for q in all_q) / n if n else 0.0
    mrr = sum(q.mrr for q in all_q) / n if n else 0.0

    return EvalReport(
        scenarios=results, k=k, hit_at_k=hit, recall_at_k=recall, mrr=mrr,
        checks_passed=checks_passed, checks_total=checks_total, n_queries=n,
        duration_s=round(time.time() - start, 3),
        embedder=engine.embedder.signature())


def run_smoke(config: Optional[Config] = None, **kw) -> EvalReport:
    """Build an engine from config (or env) and run the built-in smoke suite."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = DMemEngine(config)
    try:
        return run(engine, SMOKE_SCENARIOS, **kw)
    finally:
        engine.close()


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def format_report(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append("DMem retrieval smoke report")
    lines.append("=" * 52)
    lines.append(f"embedder      : {report.embedder}")
    if "offline-hashing" in report.embedder:
        lines.append("                (offline floor — set EMBEDDING_HOST_URL "
                     "for real quality)")
    lines.append(f"queries       : {report.n_queries}   k={report.k}   "
                 f"({report.duration_s}s)")
    lines.append("")
    lines.append(f"  hit@{report.k:<3}     : {report.hit_at_k:.2f}")
    lines.append(f"  recall@{report.k:<3}  : {report.recall_at_k:.2f}")
    lines.append(f"  MRR         : {report.mrr:.2f}")
    lines.append(f"  checks      : {report.checks_passed}/{report.checks_total} "
                 f"passed")
    lines.append("")
    for s in report.scenarios:
        lines.append(f"• {s.name}")
        for q in s.queries:
            flag = "ok " if q.hit else "MISS"
            lines.append(f"    [{flag}] hit={q.hit:.0f} recall={q.recall:.2f} "
                         f"mrr={q.mrr:.2f}  {q.query}")
        for c in s.checks:
            flag = "PASS" if c.passed else "FAIL"
            lines.append(f"    [{flag}] {c.label} — {c.detail}")
    lines.append("=" * 52)
    verdict = "PASS" if report.passed() else "FAIL"
    lines.append(f"overall: {verdict}")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(
        prog="dmem-eval",
        description="Run the DMem retrieval-quality smoke suite against your "
                    "configuration (reads env vars like the SDK).")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--k", type=int, default=None, help="top-k cutoff for metrics")
    ap.add_argument("--hit-floor", type=float, default=0.5,
                    help="minimum hit@k for an overall pass (default 0.5)")
    args = ap.parse_args()

    report = run_smoke(k=args.k)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    sys.exit(0 if report.passed(hit_floor=args.hit_floor) else 1)
