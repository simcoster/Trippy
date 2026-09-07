"""Extractor date_intent: הקרוב vs הבא, and weeks_from_now.

Few-shots were added after the 30B mapped שישי הקרוב → when=next and dropped
weeks_from_now on בעוד שבועיים. Live tests freeze today to Monday 7 Sep 2026
and run five trials each.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from source.agent.dates import resolve_dates
from source.agent.prompts import EXTRACTOR_SYSTEM_PROMPT

load_dotenv()

MONDAY = date(2026, 9, 7)
THIS_FRIDAY = date(2026, 9, 11)
NEXT_FRIDAY = date(2026, 9, 18)
WEEKEND_IN_TWO_WEEKS = date(2026, 9, 25)

PROMPT_KAROV = "בשישי הקרוב"
PROMPT_HABA = "בשישי הבא"
PROMPT_TWO_WEEKS = "סוף השבוע בעוד שבועיים"
PROMPT_Q2 = "יש מקום בחורשת טל לזוג בשישי הקרוב עד 400 שקל ללילה?"
PROMPT_Q4 = "משהו שקט במדבר לזוג, אפשר להביא כלב, סוף השבוע בעוד שבועיים"

TRIALS = range(5)


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


def _intent(constraints: dict) -> dict:
    raw = constraints.get("date_intent") or {}
    assert isinstance(raw, dict), raw
    return raw


def _as_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _date_start(constraints: dict) -> str:
    date_field = constraints.get("date")
    assert isinstance(date_field, dict), f"expected date object, got {date_field!r}"
    return str(date_field.get("start") or "")[:10]


def test_prompt_contains_karov_haba_weeks_from_now_shots():
    text = EXTRACTOR_SYSTEM_PROMPT
    assert '"weeks_from_now": N | null' in text, text
    assert '"horizon_days": N | null' in text, text
    assert "בשישי הקרוב" in text
    assert "בשישי הבא" in text
    assert "סוף השבוע בעוד שבועיים" in text
    assert '"when": "this"' in text
    assert '"weeks_from_now": 2' in text
    assert "Do not emit" in text and "when if weeks_from_now" in text


def test_resolve_this_friday_from_monday():
    resolved = resolve_dates(
        kind="weekday", weekday="friday", when="this", nights=1, today=MONDAY
    )
    assert resolved["windows"] == [
        {"start": THIS_FRIDAY.isoformat(), "end": "2026-09-12"}
    ]


def test_resolve_next_friday_from_monday():
    resolved = resolve_dates(
        kind="weekday", weekday="friday", when="next", nights=1, today=MONDAY
    )
    assert resolved["windows"] == [
        {"start": NEXT_FRIDAY.isoformat(), "end": "2026-09-19"}
    ]


def test_resolve_weekend_two_weeks_from_monday():
    resolved = resolve_dates(
        kind="weekend", weeks_from_now=2, nights=1, today=MONDAY
    )
    assert resolved["windows"] == [
        {"start": WEEKEND_IN_TWO_WEEKS.isoformat(), "end": "2026-09-26"}
    ]


def _freeze_monday(monkeypatch: pytest.MonkeyPatch) -> None:
    from source.agent import graph as agent_graph
    from source.agent.dates import resolve_dates as resolve

    monkeypatch.setattr(agent_graph, "today_il", lambda today=None: MONDAY)
    monkeypatch.setattr(
        agent_graph,
        "resolve_dates_tool",
        SimpleNamespace(invoke=lambda args: resolve(**args, today=MONDAY)),
    )


def _extract(prompt: str) -> dict:
    from source.agent.graph import extractor_node, planner_model

    assert planner_model.temperature == 0
    result = extractor_node({"messages": [HumanMessage(content=prompt)]})
    return _extractor_constraints_json(result["messages"])


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_extractor_karov_friday_is_this_week(trial, monkeypatch):
    _freeze_monday(monkeypatch)
    constraints = _extract(PROMPT_KAROV)
    intent = _intent(constraints)
    assert str(intent.get("kind") or "").lower() == "weekday", intent
    assert str(intent.get("weekday") or "").lower() == "friday", intent
    assert str(intent.get("when") or "").lower() == "this", intent
    assert _date_start(constraints) == THIS_FRIDAY.isoformat(), constraints.get("date")


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_extractor_haba_friday_is_next_week(trial, monkeypatch):
    _freeze_monday(monkeypatch)
    constraints = _extract(PROMPT_HABA)
    intent = _intent(constraints)
    assert str(intent.get("kind") or "").lower() == "weekday", intent
    assert str(intent.get("weekday") or "").lower() == "friday", intent
    assert str(intent.get("when") or "").lower() == "next", intent
    assert _date_start(constraints) == NEXT_FRIDAY.isoformat(), constraints.get("date")


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_extractor_weekend_in_two_weeks(trial, monkeypatch):
    _freeze_monday(monkeypatch)
    constraints = _extract(PROMPT_TWO_WEEKS)
    intent = _intent(constraints)
    assert str(intent.get("kind") or "").lower() == "weekend", intent
    assert _as_int(intent.get("weeks_from_now")) == 2, intent
    assert str(intent.get("when") or "").lower() != "next", intent
    assert _date_start(constraints) == WEEKEND_IN_TWO_WEEKS.isoformat(), (
        constraints.get("date"),
        intent,
    )


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_extractor_named_site_karov_friday_is_this_week(trial, monkeypatch):
    _freeze_monday(monkeypatch)
    constraints = _extract(PROMPT_Q2)
    intent = _intent(constraints)
    assert str(intent.get("weekday") or "").lower() == "friday", intent
    assert str(intent.get("when") or "").lower() == "this", intent
    assert _date_start(constraints) == THIS_FRIDAY.isoformat(), (
        constraints.get("date"),
        intent,
    )


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_extractor_desert_weekend_in_two_weeks(trial, monkeypatch):
    _freeze_monday(monkeypatch)
    constraints = _extract(PROMPT_Q4)
    intent = _intent(constraints)
    assert str(intent.get("kind") or "").lower() == "weekend", intent
    assert _as_int(intent.get("weeks_from_now")) == 2, intent
    assert str(intent.get("when") or "").lower() != "next", intent
    assert _date_start(constraints) == WEEKEND_IN_TWO_WEEKS.isoformat(), (
        constraints.get("date"),
        intent,
    )
