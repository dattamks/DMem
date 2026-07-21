# Retrieval-quality smoke harness

A fast, dependency-free way to get a **day-one signal** on retrieval quality for
*your* configuration — and to measure the lift when you plug in a real embedding
endpoint. This is deliberately lighter than formal benchmarking (LongMemEval /
LoCoMo), which remains a separate, deferred track.

## Run it

```bash
dmem-eval                 # human-readable report, reads env vars like the SDK
dmem-eval --json          # machine-readable
dmem-eval --k 5 --hit-floor 0.6
```

Exit code is `0` on pass, `1` on fail — usable as a CI gate. "Pass" means
retrieval clears the hit floor **and** every behavioral check passes.

Programmatic:

```python
from dmem.eval import run_smoke
report = run_smoke()                 # uses Config.from_env()
print(report.hit_at_k, report.mrr, report.all_checks_passed)
```

## What it measures

**Retrieval metrics** over a small labelled dataset (relevance judged by
case-insensitive substring, so it's embedder-agnostic):

| Metric | Meaning |
|---|---|
| `hit@k` | fraction of queries with any relevant item in the top-k |
| `recall@k` | fraction of a query's expected phrases found in top-k |
| `MRR` | mean reciprocal rank of the first relevant item |

**Behavioral checks** (embedder-independent, must always pass): single-valued
contradiction supersede, multi-valued accumulation, conflict surfacing, and
credential redaction.

## Reading the result

The offline hashing embedder is a **floor** — it only matches on lexical
overlap, so `MRR` in particular will be well below what a real model gives.
Point `EMBEDDING_HOST_URL` at your embedding model and re-run: the delta in
`recall@k` / `MRR` is the quality your retrieval actually gets. The report notes
when it's running on the offline floor.

Behavioral checks failing is a real regression regardless of embedder — they
assert the memory *semantics*, not ranking.

## Extending the dataset

`dmem.eval.dataset.SMOKE_SCENARIOS` is plain Python. Add a `Scenario` with
`messages` / `documents`, `Query` items (with `relevant` phrases and optional
`must_not`), and optional `checks` callables `(engine, namespace) -> (label,
passed, detail)`. Pass your own list to `dmem.eval.run(engine, scenarios)`.
