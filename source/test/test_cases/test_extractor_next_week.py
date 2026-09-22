"""Bare 'next week' / לשבוע הבא is kind=week, not tonight."""

from __future__ import annotations

import json
from datetime import date

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from source.agent.dates import resolve_dates
from source.agent.prompts import EXTRACTOR_SYSTEM_PROMPT

load_dotenv()

MONDAY = date(2026, 9, 14)
THURSDAY = date(2026, 9, 17)
PROMPT_BARE = "לשבוע הבא"
PROMPT_FULL = (
    "משהו לשבוע הבא במדבר ל3 אנשים, חשוב לנו ניקיון. אחד שומר שבת"
)
TRIALS = range(5)


def test_prompt_contains_next_week_shot():
    text = EXTRACTOR_SYSTEM_PROMPT
    assert '"kind": "weekday" | "weekend" | "on" | "week" | null' in text
    assert "לשבוע הבא" in text
    assert '"kind": "week", "when": "next"' in text
    assert "shabbat observant" in text
    assert "Not kind=\"on\"" in text or "not kind=\"on\"" in text.lower()


def test_resolve_next_week_from_monday_keeps_weekend():
    resolved = resolve_dates(kind="week", when="next", nights=1, today=MONDAY)
    assert resolved["truncated"] is False
    assert resolved["notice"] is None
    assert resolved["windows"] == [
        {"start": "2026-09-21", "end": "2026-09-22"},
        {"start": "2026-09-22", "end": "2026-09-23"},
        {"start": "2026-09-23", "end": "2026-09-24"},
        {"start": "2026-09-24", "end": "2026-09-25"},
        {"start": "2026-09-25", "end": "2026-09-26"},
        {"start": "2026-09-26", "end": "2026-09-27"},
        {"start": "2026-09-27", "end": "2026-09-28"},
    ]


def test_resolve_this_week_from_thursday_is_four_nights():
    resolved = resolve_dates(kind="week", when="this", nights=1, today=THURSDAY)
    assert resolved["truncated"] is False
    assert resolved["notice"] is None
    assert resolved["windows"] == [
        {"start": "2026-09-17", "end": "2026-09-18"},
        {"start": "2026-09-18", "end": "2026-09-19"},
        {"start": "2026-09-19", "end": "2026-09-20"},
        {"start": "2026-09-20", "end": "2026-09-21"},
    ]


def test_on_today_with_horizon_enumerates_not_one_night():
    resolved = resolve_dates(
        kind="on", on="today", horizon_days=7, nights=1, today=MONDAY
    )
    assert resolved["truncated"] is False
    assert [w["start"] for w in resolved["windows"]] == [
        "2026-09-14",
        "2026-09-15",
        "2026-09-16",
        "2026-09-17",
        "2026-09-18",
        "2026-09-19",
        "2026-09-20",
    ]


def test_week_kind_wins_over_stray_on_today():
    resolved = resolve_dates(
        kind="week",
        when="next",
        on="today",
        horizon_days=7,
        nights=1,
        today=MONDAY,
    )
    starts = [w["start"] for w in resolved["windows"]]
    assert "2026-09-14" not in starts
    assert "2026-09-25" in starts


def _freeze_today(monkeypatch: pytest.MonkeyPatch, pinned: date) -> None:
    from source.agent import graph as agent_graph

    monkeypatch.setattr(
        agent_graph, "today_il", lambda today=None, day=pinned: day
    )


def _extractor_constraints_json(messages: list) -> dict:
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        raw = msg.content
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise AssertionError(
        "extractor_node did not return a JSON constraints AIMessage; "
        f"got {[type(m).__name__ for m in messages]}"
    )


def _extract(prompt: str) -> dict:
    from source.agent.graph import extractor_model, extractor_node

    assert extractor_model.temperature == 0
    result = extractor_node({"messages": [HumanMessage(content=prompt)]})
    return _extractor_constraints_json(result["messages"])


def _intent(constraints: dict) -> dict:
    raw = constraints.get("date_intent") or {}
    assert isinstance(raw, dict), raw
    return raw


def _window_starts(constraints: dict) -> list[str]:
    windows = constraints.get("date_windows") or []
    return [str(w.get("start") or "")[:10] for w in windows]


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_extractor_bare_next_week_is_next_iso_week(trial, monkeypatch):
    _freeze_today(monkeypatch, MONDAY)
    constraints = _extract(PROMPT_BARE)
    intent = _intent(constraints)
    assert str(intent.get("kind") or "").lower() == "week", intent
    assert str(intent.get("when") or "").lower() == "next", intent
    starts = _window_starts(constraints)
    assert starts, constraints.get("date")
    assert "2026-09-14" not in starts, starts
    assert all(s >= "2026-09-21" for s in starts), starts


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_extractor_next_week_desert_shabbat(trial, monkeypatch):
    _freeze_today(monkeypatch, MONDAY)
    constraints = _extract(PROMPT_FULL)
    intent = _intent(constraints)
    assert str(intent.get("kind") or "").lower() == "week", intent
    assert str(intent.get("when") or "").lower() == "next", intent
    assert str(intent.get("on") or "").lower() not in {"today", "tonight"}, intent
    starts = _window_starts(constraints)
    assert "2026-09-14" not in starts, starts
    assert "2026-09-25" in starts, starts
    numeric = constraints.get("numeric_constraints") or []
    sizes = [
        row.get("value")
        for row in numeric
        if str(row.get("field") or "").lower() == "party_size"
    ]
    assert 3 in [int(v) for v in sizes if v is not None], numeric
