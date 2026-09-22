"""planned_exit_time is a departure clock, forwarded to QuoteParams."""

from __future__ import annotations

import json
from dataclasses import fields
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from source.agent.constraints import normalize_constraints
from source.agent.graph import planner_node
from source.agent.search.sandbox import _sandbox_quotes_for_slots
from source.price_sandbox.client import QuoteReply, QuoteRequest
from source.price_sandbox.params import QuoteParams


def test_quote_params_has_planned_exit_time():
    names = {item.name for item in fields(QuoteParams)}
    assert "planned_exit_time" in names


def test_normalize_keeps_planned_exit_time():
    out = normalize_constraints(
        {
            "planned_exit_time": "14",
            "planned_entry_time": "19:00",
            "numeric_constraints": [],
            "semantic_constraints": [],
        }
    )
    assert out["planned_exit_time"] == "14:00"
    assert out["planned_entry_time"] == "19:00"


def test_normalize_lifts_exit_time_off_numeric():
    out = normalize_constraints(
        {
            "numeric_constraints": [
                {"field": "exit_time", "operator": ">=", "value": 14}
            ],
            "semantic_constraints": [],
        }
    )
    assert out["planned_exit_time"] == "14:00"
    assert "planned_entry_time" not in out
    assert out["numeric_constraints"] == []


def test_sandbox_quote_sends_planned_exit_time(monkeypatch):
    monkeypatch.setattr(
        "source.agent.search.sandbox.sandbox_url", lambda: "http://127.0.0.1:8503"
    )
    monkeypatch.setattr(
        "source.agent.search.sandbox.sandbox_reachable", lambda **_: True
    )
    seen: list[QuoteRequest] = []

    def fake_replies(requests: list[QuoteRequest], **_kwargs):
        seen.extend(requests)
        return [
            QuoteReply(
                request_id=requests[0].request_id,
                ok=True,
                price=1.0,
                explanation="",
            )
        ]

    monkeypatch.setattr("source.agent.search.sandbox.quote_replies", fake_replies)
    _sandbox_quotes_for_slots(
        [{"campsite_id": 2, "campsite": "חורשת טל", "accommodation_type": "אוהל"}],
        party_size=2,
        rate_period="weekend_holiday",
        planned_exit_time="14:00",
    )
    assert len(seen) == 1
    assert seen[0].params == QuoteParams(
        lodging="אוהל",
        adults_num=2,
        is_weekend_or_holiday=True,
        planned_exit_time="14:00",
    )


def test_planner_forwards_planned_exit_time(monkeypatch):
    slots = MagicMock(return_value=[])
    quotes = MagicMock(return_value=[])
    monkeypatch.setattr("source.agent.search.availability.search_open_slots", slots)
    monkeypatch.setattr("source.agent.search.availability.quote_open_slots", quotes)
    planner_node(
        {
            "messages": [
                HumanMessage(content="נצא ב-14"),
                AIMessage(
                    content=json.dumps(
                        {
                            "date": {"start": "2026-08-30", "end": "2026-08-31"},
                            "numeric_constraints": [],
                            "semantic_constraints": [],
                            "planned_exit_time": "14:00",
                        },
                        ensure_ascii=False,
                    )
                ),
            ]
        }
    )
    assert "planned_exit_time" not in slots.call_args.kwargs
    assert quotes.call_args.kwargs["planned_exit_time"] == "14:00"
