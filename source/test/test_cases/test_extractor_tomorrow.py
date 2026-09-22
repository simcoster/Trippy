"""Extractor date_intent: tomorrow / מחר is the next night, not today.

Live miss 2026-09-21: English "for tomorrow" emitted on=today because the
schema only listed that relative token. Few-shot + schema token +
resolve_dates offset. Freeze today to that Monday.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

MONDAY = date(2026, 9, 21)
TOMORROW = date(2026, 9, 22)
PROMPT_LIVE = (
    "we're looking for a place for 2 adults and 2 kids for tomorrow, "
    "with pools for the kids, maybe with a fridge"
)
PROMPT_HE = (
    "מחפשים מקום למחר ל2 מבוגרים ו2 ילדים עם בריכות לילדים, אולי עם מקרר"
)
TRIALS = range(5)


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


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_extractor_english_tomorrow_is_next_night(trial, monkeypatch):
    _freeze_today(monkeypatch, MONDAY)
    constraints = _extract(PROMPT_LIVE)
    intent = _intent(constraints)
    assert str(intent.get("kind") or "").lower() == "on", intent
    assert str(intent.get("on") or "").lower() == "tomorrow", intent
    assert str(intent.get("on") or "").lower() != "today", intent
    date_field = constraints.get("date") or {}
    assert date_field.get("start") == TOMORROW.isoformat(), (
        f"date={date_field!r} date_intent={intent!r}"
    )


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_extractor_hebrew_machar_is_next_night(trial, monkeypatch):
    _freeze_today(monkeypatch, MONDAY)
    constraints = _extract(PROMPT_HE)
    intent = _intent(constraints)
    assert str(intent.get("kind") or "").lower() == "on", intent
    assert str(intent.get("on") or "").lower() == "tomorrow", intent
    date_field = constraints.get("date") or {}
    assert date_field.get("start") == TOMORROW.isoformat(), (
        f"date={date_field!r} date_intent={intent!r}"
    )
