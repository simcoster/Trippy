"""Saturday one-night date_intent: resolve_dates + extractor logs the intent."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from source.agent.dates import resolve_dates

MONDAY = date(2026, 8, 31)
PROMPT_HE_SUMMER_STARS = (
    "מקום עם מזג אוויר נחמד בקיץ שאפשר לראות בו כוכבים "
    "ואפשר להגיע בשבת בצהריים ללילה אחד עד ראשון"
)


def test_this_saturday_one_night_on_monday():
    resolved = resolve_dates(
        kind="weekday",
        weekday="saturday",
        when="this",
        nights=1,
        today=MONDAY,
    )
    assert resolved["windows"] == [{"start": "2026-09-05", "end": "2026-09-06"}]


def test_omitted_when_saturday_on_monday_is_upcoming_this_week():
    resolved = resolve_dates(
        weekday="saturday", nights=1, today=MONDAY
    )
    assert resolved["windows"] == [{"start": "2026-09-05", "end": "2026-09-06"}]


def test_horizon_without_kind_is_not_weekend():
    resolved = resolve_dates(horizon_days=30, today=MONDAY)
    assert resolved["windows"] == []
    assert resolved["truncated"] is False


def test_weekend_is_friday_night_one_night():
    resolved = resolve_dates(kind="weekend", when="this", today=MONDAY)
    assert resolved["windows"] == [{"start": "2026-09-04", "end": "2026-09-05"}]


def test_thursday_to_saturday_is_two_nights_not_weekend():
    resolved = resolve_dates(
        kind="weekday",
        weekday="thursday",
        when="this",
        nights=2,
        today=MONDAY,
    )
    assert resolved["windows"] == [{"start": "2026-09-03", "end": "2026-09-05"}]


def test_extractor_payload_includes_date_intent(monkeypatch, caplog):
    from source.agent import graph as agent_graph
    from source.agent.dates import resolve_dates as resolve

    monkeypatch.setattr(agent_graph, "today_il", lambda today=None: MONDAY)
    llm_json = {
        "date_intent": {
            "kind": "weekday",
            "weekday": "saturday",
            "when": "this",
            "nights": 1,
        },
        "campsite": None,
        "numeric_constraints": [],
        "semantic_constraints": [
            {"query": "nice summer weather"},
            {"query": "stargazing"},
        ],
    }
    fake_model = MagicMock()
    fake_model.invoke.return_value = AIMessage(content=json.dumps(llm_json))
    monkeypatch.setattr(agent_graph, "planner_model", fake_model)
    monkeypatch.setattr(
        agent_graph,
        "resolve_dates_tool",
        SimpleNamespace(invoke=lambda args: resolve(**args, today=MONDAY)),
    )

    with caplog.at_level("INFO", logger="source.agent.graph"):
        result = agent_graph.extractor_node(
            {"messages": [HumanMessage(content=PROMPT_HE_SUMMER_STARS)]}
        )
    payload = json.loads(result["messages"][0].content)
    assert payload["date_intent"] == {
        "kind": "weekday",
        "weekday": "saturday",
        "when": "this",
        "nights": 1,
    }
    assert payload["date"] == {"start": "2026-09-05", "end": "2026-09-06"}
    assert any("date_intent" in rec.message for rec in caplog.records)
