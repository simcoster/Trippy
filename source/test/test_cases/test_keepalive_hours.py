"""Interval keepalive runs from 07:00 until 23:00 Asia/Jerusalem."""

import threading
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from source.agent import keepalive as keepalive_mod
from source.agent.keepalive import (
    keepalive_hours_open,
    ping_models,
    start_model_keepalive,
)

_TZ = ZoneInfo("Asia/Jerusalem")


def _at(hour: int, minute: int) -> datetime:
    return datetime(2026, 9, 22, hour, minute, tzinfo=_TZ)


def test_keepalive_hours_are_seven_until_eleven():
    assert keepalive_hours_open(_at(6, 59)) is False
    assert keepalive_hours_open(_at(7, 0)) is True
    assert keepalive_hours_open(_at(22, 59)) is True
    assert keepalive_hours_open(_at(23, 0)) is False


def test_interval_skips_ping_outside_hours(monkeypatch):
    monkeypatch.setattr(keepalive_mod, "_started", False)
    monkeypatch.setattr(keepalive_mod, "keepalive_hours_open", lambda moment=None: False)
    seen: list[str] = []

    def _ping(**_kwargs):
        seen.append("ping")

    monkeypatch.setattr(keepalive_mod, "ping_models", _ping)
    start_model_keepalive(interval_sec=0, blocking=True)
    assert seen == []


def test_ping_runs_on_a_daemon_thread():
    flags: list[bool] = []

    class _Chat:
        model_name = "kimi"

        def bind(self, **_kwargs):
            return self

        def invoke(self, messages, config=None):
            flags.append(threading.current_thread().daemon)
            return SimpleNamespace(content="ok")

    ping_models(chats={"recommender": _Chat()})
    assert flags == [True]
