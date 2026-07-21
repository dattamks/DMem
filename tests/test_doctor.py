"""Preflight validator (dmem doctor) + the pgvector-enabled probe."""

import pytest

from dmem.doctor import (Check, run_checks, format_report, _check_embedding,
                         _check_graph, _check_pgvector)


_PRO_ENV = {
    "MEMORY_TIER": "pro",
    "PGVECTOR_URL": "postgresql://user:secret@db:5432/app",
    "GRAPH_DB": "neo4j", "GRAPH_DB_URL": "bolt://graph:7687",
    "EMBEDDING_PROVIDER": "openai",
    "EMBEDDING_HOST_URL": "https://api.openai.com/v1",
}


# -- checklist logic (no live probes) ---------------------------------------

def test_ready_when_all_prereqs_present():
    report = run_checks(_PRO_ENV, live=False)
    assert report.ready is True


def test_not_ready_when_graph_missing():
    env = dict(_PRO_ENV)
    del env["GRAPH_DB_URL"]
    report = run_checks(env, live=False)
    assert report.ready is False
    assert any("Graph database" in c.name and c.status == "fail"
               for c in report.checks)


def test_not_ready_when_pgvector_missing():
    env = dict(_PRO_ENV)
    del env["PGVECTOR_URL"]
    assert run_checks(env, live=False).ready is False


def test_not_ready_when_embedding_missing():
    env = dict(_PRO_ENV)
    del env["EMBEDDING_HOST_URL"]
    del env["EMBEDDING_PROVIDER"]
    assert run_checks(env, live=False).ready is False


def test_native_embedding_needs_key():
    c = _check_embedding({"EMBEDDING_PROVIDER": "google"})
    assert c.status == "fail" and "API_KEY" in c.detail
    c2 = _check_embedding({"EMBEDDING_PROVIDER": "google", "EMBEDDING_API_KEY": "k"})
    assert c2.status == "ok"


def test_pgvector_url_redacted_in_report():
    report = run_checks(_PRO_ENV, live=False)
    text = format_report(report)
    assert "secret" not in text  # credentials masked
    assert "READY" in text


def test_dev_tier_does_not_enforce_pro_prereqs():
    report = run_checks({"MEMORY_TIER": "sqlite"}, live=False)
    assert report.ready is True
    assert any(c.name == "Tier" and c.status == "warn" for c in report.checks)


# -- probe wiring (stubbed, no server) --------------------------------------

def test_pgvector_probe_failure_blocks():
    def stub(url):
        return False, "pgvector NOT enabled — run CREATE EXTENSION vector;"
    c = _check_pgvector(_PRO_ENV, probe=stub)
    assert c.status == "fail" and "CREATE EXTENSION" in c.detail


def test_graph_probe_success():
    def stub(kind, url, user, pw, db):
        return True, "reachable"
    c = _check_graph(_PRO_ENV, probe=stub)
    assert c.status == "ok"


def test_report_lists_gaps_and_exit_intent():
    env = dict(_PRO_ENV)
    del env["PGVECTOR_URL"]
    report = run_checks(env, live=False)
    text = format_report(report)
    assert "NOT READY" in text
    assert "prerequisites.md" in text


# -- ensure_pgvector probe logic (fake connection) --------------------------

class _FakeCur:
    def __init__(self, result):
        self._result = result
    def fetchone(self):
        return self._result


class _FakeConn:
    """Emulates psycopg's conn.execute(...).fetchone() for ensure_pgvector."""
    def __init__(self, enabled=False, available=True, can_create=True):
        self._enabled = enabled
        self._available = available
        self._can_create = can_create
        self.created = False

    def execute(self, sql, *a):
        s = sql.lower()
        if "pg_extension" in s:
            return _FakeCur((1,) if self._enabled else None)
        if "pg_available_extensions" in s:
            return _FakeCur((1,) if self._available else None)
        if "create extension" in s:
            if not self._can_create:
                raise RuntimeError("permission denied")
            self.created = True
            return _FakeCur(None)
        return _FakeCur(None)


def test_ensure_pgvector_already_enabled():
    from dmem.stores.pgvector_store import ensure_pgvector
    assert ensure_pgvector(_FakeConn(enabled=True)) == "enabled"


def test_ensure_pgvector_enables_when_available():
    from dmem.stores.pgvector_store import ensure_pgvector
    conn = _FakeConn(enabled=False, available=True)
    assert ensure_pgvector(conn) == "created"
    assert conn.created is True


def test_ensure_pgvector_not_installed_raises():
    from dmem.stores.pgvector_store import ensure_pgvector
    from dmem.errors import StoreError
    with pytest.raises(StoreError) as e:
        ensure_pgvector(_FakeConn(enabled=False, available=False))
    assert "not installed" in str(e.value).lower()


def test_ensure_pgvector_no_permission_raises_with_fix():
    from dmem.stores.pgvector_store import ensure_pgvector
    from dmem.errors import StoreError
    with pytest.raises(StoreError) as e:
        ensure_pgvector(_FakeConn(available=True, can_create=False))
    assert "CREATE EXTENSION vector" in str(e.value)
