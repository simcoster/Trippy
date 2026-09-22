"""planned_entry_time is a clock hour, not a date and not semantic RAG."""

from __future__ import annotations

from source.agent.constraints import normalize_constraints, parse_planned_entry_time
from source.agent.search import _sandbox_quotes_for_slots
from source.price_sandbox.client import QuoteReply, QuoteRequest
from source.price_sandbox.params import QuoteParams


def test_parse_planned_entry_time_hours():
    assert parse_planned_entry_time(19) == "19:00"
    assert parse_planned_entry_time("19") == "19:00"
    assert parse_planned_entry_time("19:00") == "19:00"
    assert parse_planned_entry_time("7:30") == "07:30"
    assert parse_planned_entry_time(19.0) == "19:00"
    assert parse_planned_entry_time(None) is None
    assert parse_planned_entry_time("afternoon") is None
    assert parse_planned_entry_time(24) is None


def test_normalize_keeps_planned_entry_time():
    out = normalize_constraints(
        {
            "planned_entry_time": "19",
            "numeric_constraints": [
                {"field": "party_size", "operator": ">=", "value": 4}
            ],
            "semantic_constraints": [
                {"query": "pools for children", "locus": "site"}
            ],
        }
    )
    assert out["planned_entry_time"] == "19:00"
    assert out["numeric_constraints"] == [
        {"field": "party_size", "operator": ">=", "value": 4}
    ]
    assert out["semantic_constraints"] == [
        {"query": "pools for children", "locus": "site"}
    ]


def test_normalize_lifts_entry_time_off_numeric():
    out = normalize_constraints(
        {
            "numeric_constraints": [
                {"field": "entry_time", "operator": ">=", "value": 19}
            ],
            "semantic_constraints": [],
        }
    )
    assert out["planned_entry_time"] == "19:00"
    assert out["numeric_constraints"] == []


def test_sandbox_quote_sends_planned_entry_time(monkeypatch):
    monkeypatch.setattr(
        "source.agent.search.sandbox_url", lambda: "http://127.0.0.1:8503"
    )
    monkeypatch.setattr(
        "source.agent.search.sandbox_reachable", lambda **_: True
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

    monkeypatch.setattr("source.agent.search.quote_replies", fake_replies)
    _sandbox_quotes_for_slots(
        [{"campsite_id": 2, "campsite": "חורשת טל", "accommodation_type": "אוהל"}],
        party_size=2,
        rate_period="weekday",
        planned_entry_time="19:00",
    )
    assert len(seen) == 1
    assert seen[0].params == QuoteParams(
        lodging="אוהל",
        adults_num=2,
        planned_entry_time="19:00",
    )
