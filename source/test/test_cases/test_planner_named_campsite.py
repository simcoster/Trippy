"""Named-site lookup is only for an explicit park name, then availability by hotel_id."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage

from source.agent.constraints import campsite_name_from_parsed
from source.agent.graph import planner_node


def _constraints_state(constraints: dict) -> dict:
    return {
        "messages": [
            HumanMessage(content="2 rooms in Horshat Tal"),
            AIMessage(content=json.dumps(constraints, ensure_ascii=False)),
        ]
    }


def _chat_payloads(result: dict) -> list[dict]:
    out: list[dict] = []
    for msg in result["messages"]:
        if not isinstance(msg, ChatMessage):
            continue
        try:
            data = json.loads(str(msg.content))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


@pytest.fixture
def named_site_db(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        "source.agent.graph._query_vec_literal", lambda query: "[0]"
    )
    lookup = MagicMock(
        return_value=[
            {
                "id": 1,
                "name": "חניון לילה גן לאומי חורשת טל",
                "hotel_id": 1,
                "booking_hotel_id": "9_1",
            }
        ]
    )
    availability = MagicMock(
        return_value=[
            {
                "hotel_id": 1,
                "start": "2026-08-30",
                "end": "2026-09-01",
                "adults_no": 3,
                "room_count": 2,
                "accommodation_type_id": 3,
                "accommodation_type": "בונגלו עם מזגן",
            }
        ]
    )
    campsites = MagicMock(return_value=[])
    monkeypatch.setattr("source.agent.graph.lookup_campsite_by_name", lookup)
    monkeypatch.setattr("source.agent.graph.search_availability", availability)
    monkeypatch.setattr("source.agent.graph.search_campsites", campsites)
    monkeypatch.setattr(
        "source.agent.graph.search_stated_amenities", MagicMock(return_value=[])
    )
    monkeypatch.setattr(
        "source.agent.graph.search_review_claims", MagicMock(return_value=[])
    )
    return SimpleNamespace(
        lookup=lookup, availability=availability, campsites=campsites
    )


def test_campsite_name_from_parsed_string():
    assert campsite_name_from_parsed({"campsite": "Horashat Tal"}) == "Horashat Tal"
    assert campsite_name_from_parsed({"campsite": None}) is None
    assert campsite_name_from_parsed({"campsite": {"name": "חורשת טל"}}) == "חורשת טל"


def test_planner_named_campsite_looks_up_id_then_availability(
    named_site_db: SimpleNamespace,
):
    result = planner_node(
        _constraints_state(
            {
                "date": {"start": "2026-08-30", "end": "2026-09-01"},
                "campsite": "Horshat Tal",
                "numeric_constraints": [
                    {"field": "party_size", "operator": "=", "value": 3}
                ],
                "semantic_constraints": [],
            }
        )
    )
    named_site_db.lookup.assert_called_once_with("Horshat Tal")
    named_site_db.availability.assert_called_once_with(
        1,
        date_range={"start": "2026-08-30", "end": "2026-09-01"},
        party_size=3,
    )
    named_site_db.campsites.assert_not_called()

    payload = next(p for p in _chat_payloads(result) if "availability" in p)
    assert payload["campsite"]["hotel_id"] == 1
    assert payload["availability"][0]["accommodation_type_id"] == 3
    joined = json.dumps(_chat_payloads(result), ensure_ascii=False)
    assert "parks.org.il" not in joined


def test_planner_without_named_campsite_skips_lookup(
    named_site_db: SimpleNamespace,
):
    planner_node(
        _constraints_state(
            {
                "date": None,
                "numeric_constraints": [],
                "semantic_constraints": [{"query": "air conditioning"}],
            }
        )
    )
    named_site_db.lookup.assert_not_called()
    named_site_db.availability.assert_not_called()
    named_site_db.campsites.assert_not_called()
