"""Named-site lookup filters catalog vacancies; no name searches all sites."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage

from source.agent.constraints import campsite_name_from_parsed
from source.agent.graph import planner_node

DATE = {"start": "2026-08-30", "end": "2026-09-01"}

SLOT = {
    "campsite_id": 1,
    "campsite": "חניון לילה גן לאומי חורשת טל",
    "start": "2026-08-30",
    "end": "2026-09-01",
    "room_count": 2,
    "accommodation_type_id": 3,
    "accommodation_type": "בונגלו עם מזגן",
    "max_occupancy": 4,
    "occupancy_unknown": False,
    "price_per_night": 430.0,
}


def _constraints_state(constraints: dict) -> dict:
    return {
        "messages": [
            HumanMessage(content="2 rooms in Horshat Tal"),
            AIMessage(content=json.dumps(constraints, ensure_ascii=False)),
        ]
    }


def _fits_payload(result: dict) -> dict:
    for msg in result["messages"]:
        if not isinstance(msg, ChatMessage):
            continue
        data = json.loads(str(msg.content))
        if isinstance(data, dict) and "fits" in data:
            return data
    raise AssertionError("planner did not return a fits payload")


@pytest.fixture
def named_site_db(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        "source.agent.search._query_vec_literal", lambda query: "[0]"
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
    slots = MagicMock(return_value=[dict(SLOT)])
    campsites = MagicMock(return_value=[])
    monkeypatch.setattr("source.agent.search.lookup_campsite_by_name", lookup)
    monkeypatch.setattr("source.agent.search.search_open_slots", slots)
    monkeypatch.setattr("source.agent.search.search_campsites", campsites)
    monkeypatch.setattr(
        "source.agent.search.search_stated_amenities", MagicMock(return_value=[])
    )
    monkeypatch.setattr(
        "source.agent.search.search_review_claims", MagicMock(return_value=[])
    )
    return SimpleNamespace(lookup=lookup, slots=slots, campsites=campsites)


def test_campsite_name_from_parsed_string():
    assert campsite_name_from_parsed({"campsite": "Horashat Tal"}) == "Horashat Tal"
    assert campsite_name_from_parsed({"campsite": None}) is None
    assert campsite_name_from_parsed({"campsite": {"name": "חורשת טל"}}) == "חורשת טל"


def test_planner_named_campsite_looks_up_id_then_open_slots(
    named_site_db: SimpleNamespace,
):
    result = planner_node(
        _constraints_state(
            {
                "date": DATE,
                "campsite": "Horshat Tal",
                "numeric_constraints": [
                    {"field": "party_size", "operator": "=", "value": 3}
                ],
                "semantic_constraints": [],
            }
        )
    )
    named_site_db.lookup.assert_called_once_with("Horshat Tal")
    named_site_db.slots.assert_called_once_with(
        date_range=DATE,
        site_id=1,
        party_size=3,
        numeric_constraints=[
            {"field": "party_size", "operator": "=", "value": 3}
        ],
    )
    named_site_db.campsites.assert_not_called()

    payload = _fits_payload(result)
    assert payload["fits"][0]["campsite_id"] == 1
    assert payload["fits"][0]["accommodation_type_id"] == 3
    joined = json.dumps(payload, ensure_ascii=False)
    assert "parks.org.il" not in joined


def test_planner_without_named_campsite_searches_catalog_when_dated(
    named_site_db: SimpleNamespace,
):
    planner_node(
        _constraints_state(
            {
                "date": DATE,
                "numeric_constraints": [],
                "semantic_constraints": [{"query": "air conditioning"}],
            }
        )
    )
    named_site_db.lookup.assert_not_called()
    named_site_db.slots.assert_called_once()
    assert named_site_db.slots.call_args.kwargs["site_id"] is None
    named_site_db.campsites.assert_not_called()


def test_planner_without_date_skips_lookup_and_slots(
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
    named_site_db.slots.assert_not_called()
    named_site_db.campsites.assert_not_called()
