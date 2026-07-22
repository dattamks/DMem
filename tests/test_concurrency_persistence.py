"""Concurrency (stores are called from threads by the MCP/proxy adapters) and
persistence (data must survive an engine restart)."""

import threading
import warnings

import pytest

from dmem import Config, DMemEngine, Tier
from dmem.config import EmbeddingConfig, GraphConfig, GraphKind


def _sqlite(path):
    cfg = Config(tier=Tier.SQLITE, sqlite_path=str(path), embedding=EmbeddingConfig())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DMemEngine(cfg)


def _kuzu(path):
    cfg = Config(tier=Tier.CONSOLIDATED, embedding=EmbeddingConfig(),
                 graph=GraphConfig(kind=GraphKind.KUZU, url=str(path)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DMemEngine(cfg)


def _hammer(engine, n_threads=8, per=10):
    errors = []

    def worker(i):
        try:
            for j in range(per):
                engine.ingest_message(f"I use tool{i}x{j}.", namespace="c")
                engine.retrieve("what tools does the user use?", namespace="c")
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_concurrent_ingest_retrieve_sqlite(tmp_path):
    engine = _sqlite(tmp_path / "c.db")
    try:
        errors = _hammer(engine)
        assert not errors, errors
        uses = [f for f in engine.store.iter_facts()
                if f.predicate == "uses" and f.namespace == "c"]
        assert len(uses) == 80  # 8 threads * 10, all distinct multi-valued facts
    finally:
        engine.close()


def test_persistence_sqlite(tmp_path):
    p = tmp_path / "persist.db"
    e1 = _sqlite(p)
    e1.ingest_message("My name is Ada Lovelace.", namespace="p")
    e1.close()

    e2 = _sqlite(p)  # reopen the same file
    try:
        results = e2.retrieve("what is the user's name?", namespace="p")
        assert any("ada lovelace" in r.text.lower() for r in results)
    finally:
        e2.close()


def test_concurrent_ingest_retrieve_kuzu(tmp_path):
    pytest.importorskip("kuzu")
    engine = _kuzu(tmp_path / "c.kz")
    try:
        errors = _hammer(engine, n_threads=6, per=6)
        assert not errors, errors
        uses = [f for f in engine.store.iter_facts()
                if f.predicate == "uses" and f.namespace == "c"]
        assert len(uses) == 36
    finally:
        engine.close()


def test_persistence_kuzu(tmp_path):
    pytest.importorskip("kuzu")
    p = tmp_path / "persist.kz"
    e1 = _kuzu(p)
    e1.ingest_message("My name is Grace Hopper.", namespace="p")
    e1.close()

    e2 = _kuzu(p)  # reopen the same embedded DB
    try:
        results = e2.retrieve("what is the user's name?", namespace="p")
        assert any("grace hopper" in r.text.lower() for r in results)
    finally:
        e2.close()
