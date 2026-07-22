"""Token-savings measurement harness.

Measures the *actual* context tokens sent to an LLM WITH vs WITHOUT DMem across
the three scenarios the product targets:

- **documents**   — retrieve relevant chunks vs. stuffing the whole document
- **long convo**  — compact old turns vs. resending the full transcript
- **model switch**— a budgeted handoff vs. resending the full prior context

Savings is a *ratio* of two token counts over similar text, which is essentially
tokenizer-invariant — so the percentage is robust even though we can't fetch a
real BPE vocab offline (tiktoken is used when available, else a ~4-chars/token
approximation). Absolute counts are approximate; the % is the headline.

This measures *context volume* (what hits the LLM), which is set by the
mechanisms (top-k, budget, compaction), not by embedding quality — so it's
meaningful even with the offline embedder.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

from ..adapters.proxy import MemoryChatProxy, ProxyConfig, _content_str
from ..config import Config, Tier
from ..config import EmbeddingConfig
from ..engine import DMemEngine


def _make_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda t: len(enc.encode(t)), "tiktoken/cl100k_base"
    except Exception:
        # ~4 chars/token; ratio-robust even if absolute counts are approximate.
        return lambda t: max(1, round(len(t) / 4)), "approx(4 chars/token)"


count_tokens, TOKENIZER = _make_counter()


@dataclass
class ScenarioSaving:
    name: str
    baseline_tokens: int
    dmem_tokens: int
    detail: str = ""

    @property
    def saving_pct(self) -> float:
        if self.baseline_tokens <= 0:
            return 0.0
        return 100.0 * (1 - self.dmem_tokens / self.baseline_tokens)


@dataclass
class SavingsReport:
    scenarios: list[ScenarioSaving] = field(default_factory=list)
    tokenizer: str = TOKENIZER

    def blended_pct(self) -> float:
        b = sum(s.baseline_tokens for s in self.scenarios)
        d = sum(s.dmem_tokens for s in self.scenarios)
        return 100.0 * (1 - d / b) if b else 0.0


# --------------------------------------------------------------------------- #

_RUNBOOK_SECTIONS = [
    ("Overview", "This runbook documents the production deployment of the "
     "billing service, its dependencies, on-call procedures, and recovery steps. "
     "It is maintained by the platform team and reviewed quarterly."),
    ("Architecture", "The billing service is a Python application behind an ALB. "
     "It reads and writes a PostgreSQL 16 primary with two read replicas, uses "
     "Redis for rate limiting, and publishes events to Kafka. All traffic is "
     "TLS-terminated at the load balancer."),
    ("Hosting", "Production runs in AWS us-east-1 across three availability "
     "zones. Staging runs in us-west-2. Infrastructure is managed with Terraform "
     "and deploys are gated by CI."),
    ("Scaling", "The service autoscales between 4 and 40 tasks based on CPU and "
     "request latency. Database connections are pooled via PgBouncer with a max "
     "of 200 server connections."),
    ("On-call", "The primary on-call rotation is weekly. Pages escalate to the "
     "platform team lead after 15 minutes without acknowledgement. Runbook links "
     "are attached to every alert."),
    ("Backups", "Nightly base backups plus continuous WAL archiving give a "
     "recovery point objective of five minutes. Snapshots are retained for 30 "
     "days in an encrypted S3 bucket in a separate account."),
    ("Incidents", "For a database failover, promote the healthiest replica, "
     "update the connection secret, and restart the service tasks in a rolling "
     "fashion. Post-incident reviews are required within 48 hours."),
    ("Contacts", "Escalation goes platform lead, then engineering manager, then "
     "the director on-call. Vendor support for the database is available 24/7 "
     "under the enterprise plan."),
]


def _dev_engine() -> DMemEngine:
    import tempfile
    cfg = Config(tier=Tier.SQLITE,
                 sqlite_path=tempfile.mktemp(suffix=".db"),
                 embedding=EmbeddingConfig())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DMemEngine(cfg)


def _document_scenario(engine: DMemEngine) -> ScenarioSaving:
    ns = "tok_doc"
    engine.forget(ns)
    # a multi-section document repeated to a realistic size
    md = "# Billing Service Runbook\n\n"
    for _ in range(3):  # ~24 sections total
        for title, body in _RUNBOOK_SECTIONS:
            md += f"## {title}\n\n{body}\n\n"
    engine.ingest_document(md, document_id="runbook", namespace=ns)
    doc_tokens = count_tokens(md)

    queries = [
        "where is production hosted?",
        "what is the backup retention and RPO?",
        "how does the on-call escalation work?",
        "how does the service scale?",
        "what happens during a database failover?",
    ]
    baseline = dmem = 0
    for q in queries:
        qn = count_tokens(q)
        baseline += doc_tokens + qn          # naive: whole doc every turn
        hits = engine.retrieve(q, namespace=ns, kinds=("chunk",))
        retrieved = "\n".join(h.text for h in hits)
        dmem += count_tokens(retrieved) + qn  # DMem: only relevant chunks
    return ScenarioSaving("documents", baseline, dmem,
                          f"{len(queries)} queries over a {doc_tokens}-token doc")


def _long_conversation_scenario(engine: DMemEngine) -> ScenarioSaving:
    ns = "tok_conv"
    engine.forget(ns)
    proxy = MemoryChatProxy(engine, ProxyConfig(
        compact_over_tokens=600, keep_last_turns=6, inject_memory=False))
    # build a long transcript
    msgs = [{"role": "system", "content": "You are a helpful engineering assistant."}]
    for i in range(40):
        msgs.append({"role": "user",
                     "content": f"Question {i}: can you explain step {i} of the "
                                f"migration plan and any risks involved in detail?"})
        msgs.append({"role": "assistant",
                     "content": f"Step {i}: here is a fairly detailed explanation "
                                f"covering the approach, the rollback path, and "
                                f"the main risks to watch for during step {i}."})
    baseline = sum(count_tokens(_content_str(m)) for m in msgs)
    compacted = proxy._maybe_compact(msgs, ns)
    dmem = sum(count_tokens(_content_str(m)) for m in compacted)
    return ScenarioSaving("long_conversation", baseline, dmem,
                          f"{len(msgs)} messages -> {len(compacted)} after compaction")


def _model_switch_scenario(engine: DMemEngine) -> ScenarioSaving:
    ns = "tok_switch"
    engine.forget(ns)
    # a prior conversation that established facts + context
    prior_msgs = [
        {"role": "user", "content": "My name is Ada Lovelace. I work at "
         "Analytical Engines Inc. I prefer Rust and I use Postgres and Redis."},
        {"role": "assistant", "content": "Noted, Ada."},
    ]
    for i in range(20):
        prior_msgs.append({"role": "user",
                           "content": f"We also discussed design topic {i} at "
                                      f"length with several tradeoffs and a "
                                      f"decision recorded for topic {i}."})
        prior_msgs.append({"role": "assistant",
                           "content": f"Summary of topic {i} and the decision."})
    for m in prior_msgs:
        if m["role"] == "user":
            engine.ingest_message(m["content"], namespace=ns)

    baseline = sum(count_tokens(m["content"]) for m in prior_msgs)  # resend all
    handoff = engine.handoff("continue where we left off", namespace=ns)
    dmem = count_tokens(handoff.text)
    return ScenarioSaving("model_switch", baseline, dmem,
                          f"{len(prior_msgs)}-message history -> {handoff.token_estimate}-tok handoff")


def measure_token_savings(engine: Optional[DMemEngine] = None) -> SavingsReport:
    own = engine is None
    engine = engine or _dev_engine()
    try:
        report = SavingsReport(scenarios=[
            _document_scenario(engine),
            _long_conversation_scenario(engine),
            _model_switch_scenario(engine),
        ])
        return report
    finally:
        if own:
            engine.close()


def format_savings_report(report: SavingsReport) -> str:
    lines = ["DMem token-savings measurement",
             f"(tokenizer: {report.tokenizer}; % is ratio-robust)",
             "=" * 62,
             f"{'scenario':<20}{'baseline':>10}{'with DMem':>12}{'saving':>10}"]
    for s in report.scenarios:
        lines.append(f"{s.name:<20}{s.baseline_tokens:>10}{s.dmem_tokens:>12}"
                     f"{s.saving_pct:>9.1f}%")
        lines.append(f"    {s.detail}")
    lines.append("-" * 62)
    lines.append(f"{'blended':<20}"
                 f"{sum(s.baseline_tokens for s in report.scenarios):>10}"
                 f"{sum(s.dmem_tokens for s in report.scenarios):>12}"
                 f"{report.blended_pct():>9.1f}%")
    lines.append("=" * 62)
    lines.append("Note: measures context tokens sent to the LLM. Excludes DMem's "
                 "own overhead (embedding calls on ingest, injected-memory block),"
                 " which reduces NET savings — see docs.")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI entry
    print(format_savings_report(measure_token_savings()))


if __name__ == "__main__":  # pragma: no cover
    main()
