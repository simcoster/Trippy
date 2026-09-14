"""Production `connect()` pins `extensions` so `::vector` resolves."""

from db.connect import (
    EXPERIMENTS_OPTIONS,
    PRODUCTION_OPTIONS,
    SCHEMA_ENV,
    connect,
)


def test_production_connect_pins_extensions_search_path(monkeypatch):
    monkeypatch.delenv(SCHEMA_ENV, raising=False)
    seen: dict = {}

    def fake_connect(url, **kwargs):
        seen["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("db.connect.psycopg.connect", fake_connect)
    connect("postgresql://trippy@localhost/trippy")
    assert seen["kwargs"]["options"] == PRODUCTION_OPTIONS


def test_experiments_connect_still_wins(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "experiments")
    seen: dict = {}

    def fake_connect(url, **kwargs):
        seen["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("db.connect.psycopg.connect", fake_connect)
    connect("postgresql://trippy@localhost/trippy")
    assert seen["kwargs"]["options"] == EXPERIMENTS_OPTIONS


def test_explicit_options_are_kept(monkeypatch):
    monkeypatch.delenv(SCHEMA_ENV, raising=False)
    seen: dict = {}

    def fake_connect(url, **kwargs):
        seen["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("db.connect.psycopg.connect", fake_connect)
    connect("postgresql://trippy@localhost/trippy", options="-csearch_path=other")
    assert seen["kwargs"]["options"] == "-csearch_path=other"
