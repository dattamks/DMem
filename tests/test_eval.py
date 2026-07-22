"""Tests for the retrieval-quality smoke harness itself (offline)."""

import warnings

from dmem import Config, Tier
from dmem.config import EmbeddingConfig
from dmem.eval import run_smoke, SMOKE_SCENARIOS
from dmem.eval.dataset import Query
from dmem.eval.harness import _score_query, format_report


def _cfg(tmp_path):
    return Config(tier=Tier.SQLITE, sqlite_path=str(tmp_path / "eval.db"),
                  embedding=EmbeddingConfig())


def test_score_query_hit_and_recall():
    q = Query("where does the user work?", ["analytical engines"])
    r = _score_query(["user works_at Analytical Engines", "user prefers rust"], q)
    assert r.hit == 1.0
    assert r.recall == 1.0
    assert r.mrr == 1.0  # relevant item is first


def test_score_query_miss():
    q = Query("x", ["nonexistent phrase"])
    r = _score_query(["something else", "another thing"], q)
    assert r.hit == 0.0
    assert r.recall == 0.0
    assert r.mrr == 0.0


def test_score_query_mrr_rank_two():
    q = Query("x", ["target"])
    r = _score_query(["noise", "the target is here"], q)
    assert r.mrr == 0.5


def test_score_query_must_not_flags_leak():
    q = Query("x", ["ok"], must_not=["secret"])
    r = _score_query(["this is ok", "leaked secret value"], q)
    assert r.clean == 0.0


def test_run_smoke_offline_behavioral_checks_pass(tmp_path):
    # Behavioral checks (contradiction, multi-valued, credential safety) are
    # embedder-independent and must pass even on the offline floor.
    report = run_smoke(_cfg(tmp_path))
    assert report.checks_total >= 4
    assert report.all_checks_passed, [
        (c.label, c.detail) for s in report.scenarios for c in s.checks
        if not c.passed]


def test_run_smoke_reports_metrics(tmp_path):
    report = run_smoke(_cfg(tmp_path))
    assert report.n_queries >= 5
    assert 0.0 <= report.hit_at_k <= 1.0
    assert 0.0 <= report.recall_at_k <= 1.0
    assert "offline-hashing" in report.embedder
    # even the lexical floor should retrieve *some* of the labelled facts
    assert report.hit_at_k > 0.0


def test_report_is_idempotent_across_runs(tmp_path):
    cfg = _cfg(tmp_path)
    r1 = run_smoke(cfg)
    r2 = run_smoke(cfg)  # namespaces are wiped per run -> stable
    assert r1.hit_at_k == r2.hit_at_k
    assert r1.checks_passed == r2.checks_passed


def test_format_report_renders(tmp_path):
    report = run_smoke(_cfg(tmp_path))
    text = format_report(report)
    assert "DMem retrieval smoke report" in text
    assert "overall:" in text
    assert "hit@" in text


def test_to_dict_serializable(tmp_path):
    import json
    report = run_smoke(_cfg(tmp_path))
    json.dumps(report.to_dict())  # must not raise
