"""Planner query embeddings run concurrently, capped at five."""

from __future__ import annotations

import threading
import time

import pytest

from source.agent.search import embed


def _fake_vec(_texts: list[str], **_kwargs):
    return [[0.1, 0.2, 0.3]]


def test_query_vec_literals_uses_query_vec_literal(monkeypatch):
    monkeypatch.setattr(
        "source.agent.search.embed._query_vec_literal", lambda query: f"[{query}]"
    )
    out = embed._query_vec_literals(["quiet", "shade", "quiet"])
    assert out == {"quiet": "[quiet]", "shade": "[shade]"}


def test_multiple_query_statements_embed_in_parallel(monkeypatch):
    in_flight = 0
    peak = 0
    lock = threading.Lock()
    gate = threading.Barrier(5)

    def fake_embed(texts, **_kwargs):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            gate.wait(timeout=2)
        except threading.BrokenBarrierError:
            pytest.fail("query embeddings did not overlap")
        with lock:
            in_flight -= 1
        return _fake_vec(texts)

    monkeypatch.setattr(embed._claims_embedder, "embed", fake_embed)
    queries = [f"q{i}" for i in range(5)]
    out = embed._query_vec_literals(queries)
    assert peak == 5
    assert list(out) == queries


def test_query_embed_concurrency_caps_at_five(monkeypatch):
    in_flight = 0
    peak = 0
    lock = threading.Lock()
    release = threading.Event()
    entered = threading.Semaphore(0)

    def fake_embed(texts, **_kwargs):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        entered.release()
        assert release.wait(timeout=2)
        with lock:
            in_flight -= 1
        return _fake_vec(texts)

    monkeypatch.setattr(embed._claims_embedder, "embed", fake_embed)
    queries = [f"q{i}" for i in range(8)]
    thread = threading.Thread(target=lambda: embed._query_vec_literals(queries))
    thread.start()
    for _ in range(5):
        assert entered.acquire(timeout=2)
    time.sleep(0.05)
    with lock:
        assert peak == 5
        assert in_flight == 5
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert peak == 5
