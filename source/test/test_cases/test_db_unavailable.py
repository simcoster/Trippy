"""Postgres down fails in seconds with a Docker hint, not a multi-minute hang."""

from unittest.mock import MagicMock

import psycopg
import pytest

from db.connect import (
    DEFAULT_CONNECT_TIMEOUT,
    DatabaseUnavailable,
    connect,
    ping,
    unavailable_message,
)


def test_connect_sets_a_short_timeout(monkeypatch):
    seen: dict = {}

    def fake_connect(url, **kwargs):
        seen["url"] = url
        seen.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("db.connect.psycopg.connect", fake_connect)
    monkeypatch.setenv("DATABASE_URL", "postgresql://trippy@localhost/trippy")
    connect()
    assert seen["connect_timeout"] == DEFAULT_CONNECT_TIMEOUT


def test_connect_keeps_an_explicit_timeout(monkeypatch):
    seen: dict = {}

    def fake_connect(url, **kwargs):
        seen.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("db.connect.psycopg.connect", fake_connect)
    monkeypatch.setenv("DATABASE_URL", "postgresql://trippy@localhost/trippy")
    connect(connect_timeout=10)
    assert seen["connect_timeout"] == 10


def test_connect_wraps_operational_error(monkeypatch):
    def fake_connect(url, **kwargs):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr("db.connect.psycopg.connect", fake_connect)
    monkeypatch.setenv("DATABASE_URL", "postgresql://trippy@localhost/trippy")
    with pytest.raises(DatabaseUnavailable, match="docker compose up -d"):
        connect()


def test_ping_runs_select_1(monkeypatch):
    inner = MagicMock()
    ctx = MagicMock()
    ctx.__enter__.return_value = inner
    ctx.__exit__.return_value = False

    def fake_connect(url, **kwargs):
        return ctx

    monkeypatch.setattr("db.connect.psycopg.connect", fake_connect)
    monkeypatch.setenv("DATABASE_URL", "postgresql://trippy@localhost/trippy")
    ping()
    inner.execute.assert_called_once_with("SELECT 1")


def test_unavailable_message_names_compose():
    text = unavailable_message("connection refused")
    assert "docker compose up -d" in text
    assert "connection refused" in text
