"""Date intent parsing, resolve_dates, one availability search for every window."""

from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage

from source.agent.dates import (
    DATE_TRUNCATED_NOTICE,
    MAX_DATE_WINDOWS,
    resolve_dates,
    weekday_next_iso_week,
    weekday_this_iso_week,
)
from source.agent.graph import extractor_node, planner_node

MONDAY = date(2026, 8, 31)


def test_next_thursday_on_monday_is_ten_days_not_three():
    this_thu = weekday_this_iso_week(MONDAY, 3)
    next_thu = weekday_next_iso_week(MONDAY, 3)
    assert (this_thu - MONDAY).days == 3
    assert (next_thu - MONDAY).days == 10
    resolved = resolve_dates(
        kind="weekday", weekday="thursday", when="next", nights=1, today=MONDAY
    )
    assert resolved["windows"] == [{"start": "2026-09-10", "end": "2026-09-11"}]
    assert resolved["truncated"] is False


def test_this_thursday_on_monday_is_three_days():
    resolved = resolve_dates(
        kind="weekday", weekday="thursday", when="this", nights=1, today=MONDAY
    )
    assert resolved["windows"] == [{"start": "2026-09-03", "end": "2026-09-04"}]


def test_next_wednesday_two_nights():
    resolved = resolve_dates(
        kind="weekday",
        weekday="wednesday",
        when="next",
        nights=2,
        today=MONDAY,
    )
    assert resolved["windows"] == [{"start": "2026-09-09", "end": "2026-09-11"}]


def test_tuesday_in_three_weeks():
    resolved = resolve_dates(
        kind="weekday",
        weekday="tuesday",
        weeks_from_now=3,
        today=MONDAY,
    )
    assert resolved["windows"] == [{"start": "2026-09-22", "end": "2026-09-23"}]


def test_weekend_coming_month_caps_at_four_and_notices():
    resolved = resolve_dates(
        kind="weekend", horizon_days=160, today=MONDAY
    )
    assert resolved["truncated"] is True
    assert resolved["notice"] == DATE_TRUNCATED_NOTICE
    assert len(resolved["windows"]) == MAX_DATE_WINDOWS
    starts = [w["start"] for w in resolved["windows"]]
    assert starts == [
        (date(2026, 9, 4) + timedelta(days=7 * i)).isoformat()
        for i in range(MAX_DATE_WINDOWS)
    ]
    for window in resolved["windows"]:
        assert window["end"] > window["start"]


def test_weekend_horizon_intent():
    resolved = resolve_dates(
        kind="weekend", horizon_days=30, nights=2, today=MONDAY
    )
    assert resolved["truncated"] is False
    assert len(resolved["windows"]) == 4
    assert resolved["windows"][0] == {"start": "2026-09-04", "end": "2026-09-06"}


SLOT = {
    "campsite_id": 3,
    "campsite": "Park A",
    "start": "2026-09-04",
    "end": "2026-09-06",
    "room_count": 1,
    "accommodation_type_id": 11,
    "accommodation_type": "בונגלו",
    "max_occupancy": 4,
    "occupancy_unknown": False,
    "price_per_night": 400.0,
}


def _constraints_state(constraints: dict) -> dict:
    return {
        "messages": [
            HumanMessage(content="plan a trip"),
            AIMessage(content=json.dumps(constraints, ensure_ascii=False)),
        ]
    }


def _fits_payload(result: dict) -> dict:
    msg = result["messages"][0]
    assert isinstance(msg, ChatMessage)
    return json.loads(str(msg.content))


def _slots_for_windows(**kwargs):
    windows = kwargs.get("date_windows")
    if not windows:
        windows = [kwargs["date_range"]]
    return [
        {**SLOT, "start": window["start"], "end": window["end"]}
        for window in windows
    ]


@pytest.fixture
def db_searches(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        "source.agent.search.embed._query_vec_literal", lambda query: "[0]"
    )
    slots = MagicMock(side_effect=_slots_for_windows)
    monkeypatch.setattr("source.agent.search.availability.search_open_slots", slots)
    monkeypatch.setattr(
        "source.agent.search.amenities.search_stated_amenities", MagicMock(return_value=[])
    )
    monkeypatch.setattr(
        "source.agent.search.claims.search_review_claims", MagicMock(return_value=[])
    )
    monkeypatch.setattr(
        "source.agent.search.campsites.lookup_campsite_by_name", MagicMock(return_value=[])
    )
    return SimpleNamespace(slots=slots)


def test_planner_searches_all_windows_at_once(db_searches: SimpleNamespace):
    w1 = {"start": "2026-09-04", "end": "2026-09-06"}
    w2 = {"start": "2026-09-11", "end": "2026-09-13"}
    result = planner_node(
        _constraints_state(
            {
                "date": w1,
                "date_windows": [w1, w2],
                "numeric_constraints": [],
                "semantic_constraints": [],
            }
        )
    )
    assert db_searches.slots.call_count == 1
    assert db_searches.slots.call_args.kwargs["date_windows"] == [w1, w2]
    payload = _fits_payload(result)
    assert len(payload["fits"]) == 1
    assert payload["fits"][0]["start"] == "2026-09-04"
    assert [d["start"] for d in payload["fits"][0]["dates"]] == [
        "2026-09-04",
        "2026-09-11",
    ]
    assert isinstance(payload["open_slots_query"], dict)


def test_planner_caps_windows_at_four(db_searches: SimpleNamespace):
    windows = [
        {
            "start": (date(2026, 9, 4) + timedelta(days=7 * i)).isoformat(),
            "end": (date(2026, 9, 6) + timedelta(days=7 * i)).isoformat(),
        }
        for i in range(MAX_DATE_WINDOWS + 1)
    ]
    result = planner_node(
        _constraints_state(
            {
                "date": windows[0],
                "date_windows": windows,
                "date_truncated": True,
                "date_notice": DATE_TRUNCATED_NOTICE,
                "numeric_constraints": [],
                "semantic_constraints": [],
            }
        )
    )
    assert db_searches.slots.call_count == 1
    assert (
        len(db_searches.slots.call_args.kwargs["date_windows"]) == MAX_DATE_WINDOWS
    )
    payload = _fits_payload(result)
    assert payload["date_notice"] == DATE_TRUNCATED_NOTICE
    assert len(payload["fits"]) == 1
    assert [d["start"] for d in payload["fits"][0]["dates"]] == [
        w["start"] for w in windows[:MAX_DATE_WINDOWS]
    ]


def test_extractor_calls_resolve_dates_tool(monkeypatch: pytest.MonkeyPatch):
    from source.agent import graph as agent_graph

    monkeypatch.setattr(agent_graph, "today_il", lambda today=None: MONDAY)

    llm_json = {
        "date_intent": {
            "kind": "weekday",
            "weekday": "thursday",
            "when": "next",
            "nights": 2,
        },
        "campsite": None,
        "numeric_constraints": [
            {"field": "party_size", "operator": "=", "value": 3}
        ],
        "semantic_constraints": [{"query": "air conditioning"}],
    }
    fake_model = MagicMock()
    fake_model.invoke.return_value = AIMessage(content=json.dumps(llm_json))
    monkeypatch.setattr(agent_graph, "extractor_model", fake_model)

    result = extractor_node(
        {
            "messages": [
                HumanMessage(
                    content="משהו עם מזגן ל3 מבוגרים ביום חמישי הבא ל2 לילות"
                )
            ]
        }
    )
    payload = json.loads(result["messages"][0].content)
    assert payload["date"] == {"start": "2026-09-10", "end": "2026-09-12"}
    assert payload["date_windows"] == [payload["date"]]
    assert payload["date_truncated"] is False


def test_extractor_truncation_notice_from_tool(monkeypatch: pytest.MonkeyPatch):
    from source.agent import graph as agent_graph

    monkeypatch.setattr(agent_graph, "today_il", lambda today=None: MONDAY)
    llm_json = {
        "date_intent": {"kind": "weekend", "horizon_days": 160, "nights": 2},
        "campsite": None,
        "numeric_constraints": [],
        "semantic_constraints": [],
    }
    fake_model = MagicMock()
    fake_model.invoke.return_value = AIMessage(content=json.dumps(llm_json))
    monkeypatch.setattr(agent_graph, "extractor_model", fake_model)
    result = extractor_node(
        {"messages": [HumanMessage(content="סופ״ש בחודש הקרוב")]}
    )
    payload = json.loads(result["messages"][0].content)
    assert payload["date_truncated"] is True
    assert payload["date_notice"] == DATE_TRUNCATED_NOTICE
    assert len(payload["date_windows"]) == MAX_DATE_WINDOWS
