"""Availability table env, frozen snapshot identifier, pinned today."""

from __future__ import annotations

import pytest

from db.experiments import table_name
from source.agent.dates import today_il
from source.agent.search import _availability_relation, _open_slots_sql


def test_open_slots_sql_uses_frozen_table(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRIPPY_AVAILABILITY_TABLE", "availability_frozen")
    sql, _params = _open_slots_sql(
        date_range={"start": "2026-09-17", "end": "2026-09-18"},
        site_id=None,
        party_size=None,
        limit=10,
    )
    assert "FROM availability_frozen a" in sql
    assert "FROM availability a\n" not in sql


def test_availability_relation_rejects_injection():
    with pytest.raises(ValueError):
        table_name("availability; drop table campsites")


def test_default_availability_relation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TRIPPY_AVAILABILITY_TABLE", raising=False)
    assert _availability_relation() == "availability"


def test_today_il_reads_trippy_today(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRIPPY_TODAY", "2026-09-08")
    assert today_il().isoformat() == "2026-09-08"
