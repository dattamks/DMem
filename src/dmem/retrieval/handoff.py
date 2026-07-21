"""Token-budgeted, priority-ordered handoff builder.

On a model switch, produce a compact context handoff:
- Hard token budget. If over budget, the LOWEST-priority items drop first
  (never arbitrary truncation).
- Priority combines fact-type importance, confidence, and retrieval score.
- Conflicts are surfaced explicitly in the output, not silently collapsed.
- If retrieval was empty or everything scored below the confidence threshold,
  the handoff is flagged ``low_confidence`` so the engine triggers the hard
  fallback (escape hatch / surface the gap) instead of proceeding with nothing.

Credential-typed facts are excluded from the handoff text by default — secrets
should not travel in a context blob — but their existence is noted.
"""

from __future__ import annotations

from typing import Optional

from ..types import FACT_TYPE_PRIORITY, FactType, Handoff, RetrievalResult


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _priority(r: RetrievalResult) -> float:
    if r.kind == "fact":
        ft = FactType.coerce(r.payload.get("fact_type"))
        base = FACT_TYPE_PRIORITY.get(ft, 30)
        conf = float(r.payload.get("confidence", 1.0))
    else:
        base = 55  # document chunks: moderate priority
        conf = 1.0
    # blend type priority, confidence, and retrieval score
    return base + conf * 5 + r.score * 10


def build_handoff(
    results: list[RetrievalResult], *, token_budget: int,
    conflicts: Optional[list[dict]] = None,
    low_confidence_threshold: float = 0.15,
    include_credentials: bool = False,
) -> Handoff:
    conflicts = conflicts or []

    # Confidence gate: if nothing cleared the bar, flag for hard fallback.
    top_score = max((r.score for r in results), default=0.0)
    low_conf = (not results) or (top_score < low_confidence_threshold)

    ranked = sorted(results, key=_priority, reverse=True)

    header = "Context handoff (compact):"

    # Build the conflict section first (capped) so its cost is reserved against
    # the same budget — the whole handoff text must fit, not just the fact lines.
    conflict_lines: list[str] = []
    if conflicts:
        conflict_lines.append("\nChanged facts (surfaced, not resolved):")
        for c in conflicts[:3]:
            vals = " -> ".join(str(v["object"]) for v in c["values"][:4])
            conflict_lines.append(f"- {c['subject']} {c['predicate']}: {vals}"[:160])
    conflict_cost = _approx_tokens("\n".join(conflict_lines)) if conflict_lines else 0

    lines: list[str] = []
    included: list[RetrievalResult] = []
    dropped: list[RetrievalResult] = []
    used = 0
    budget = token_budget - _approx_tokens(header) - conflict_cost

    for r in ranked:
        if r.kind == "fact" and FactType.coerce(r.payload.get("fact_type")) \
                is FactType.CREDENTIAL and not include_credentials:
            # note existence but don't emit the secret-bearing value
            dropped.append(r)
            continue
        line = _format_line(r)
        cost = _approx_tokens(line)
        if used + cost > budget and included:
            dropped.append(r)
            continue
        lines.append(line)
        included.append(r)
        used += cost

    body_parts = [header]
    if lines:
        body_parts.append("\n".join(lines))
    else:
        body_parts.append("(no high-confidence context found)")
    body_parts.extend(conflict_lines)

    cred_dropped = [r for r in dropped if r.kind == "fact"
                    and FactType.coerce(r.payload.get("fact_type")) is FactType.CREDENTIAL]
    notes: list[str] = []
    if cred_dropped:
        notes.append(f"{len(cred_dropped)} credential fact(s) withheld from handoff.")
    if dropped and len(dropped) > len(cred_dropped):
        notes.append(f"{len(dropped) - len(cred_dropped)} lower-priority item(s) "
                     "dropped to fit token budget.")
    if low_conf:
        notes.append("Low/empty confidence — trigger escape hatch or surface gap.")

    text = "\n".join(body_parts).strip()
    return Handoff(
        text=text, token_estimate=_approx_tokens(text), included=included,
        dropped=dropped, conflicts=conflicts, low_confidence=low_conf, notes=notes)


def _format_line(r: RetrievalResult) -> str:
    if r.kind == "fact":
        marker = "!" if r.conflict else "-"
        return f"{marker} {r.text}"
    src = r.payload.get("section") or r.payload.get("concept") or "doc"
    return f"- [{src}] {r.text}"
