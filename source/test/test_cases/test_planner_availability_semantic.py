"""Two-stage planner: vacancies + prices, then in-set semantic intersection."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage

from source.agent.graph import (
    _price_matches,
    _price_per_night_constraint,
    _rate_period_for_stay,
    planner_node,
)

FAKE_VEC = "[0.1,0.2,0.3]"

DATE = {"start": "2026-08-30", "end": "2026-09-01"}

SLOT_AC = {
    "campsite_id": 3,
    "campsite": "Park A",
    "start": "2026-08-30",
    "end": "2026-09-01",
    "room_count": 1,
    "accommodation_type_id": 11,
    "accommodation_type": "בונגלו עם מזגן",
    "max_occupancy": 4,
    "occupancy_unknown": False,
    "price_per_night": 400.0,
}
SLOT_TENT = {
    "campsite_id": 4,
    "campsite": "Park B",
    "start": "2026-08-30",
    "end": "2026-09-01",
    "room_count": 1,
    "accommodation_type_id": 22,
    "accommodation_type": "אוהל",
    "max_occupancy": 6,
    "occupancy_unknown": False,
    "price_per_night": 180.0,
}


def _constraints_state(constraints: dict) -> dict:
    return {
        "messages": [
            HumanMessage(content="place for 3 people, 2 nights, with AC"),
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
def two_stage(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        "source.agent.search._query_vec_literal", lambda query: FAKE_VEC
    )
    slots = MagicMock(return_value=[dict(SLOT_AC), dict(SLOT_TENT)])
    amenities = MagicMock(
        return_value=[
            {
                "amenity": "air_conditioning",
                "accommodation_type_id": 11,
                "accommodation_type": SLOT_AC["accommodation_type"],
                "hotel_id": 3,
                "distance": -0.9,
            }
        ]
    )
    claims = MagicMock(return_value=[])
    lookup = MagicMock(return_value=[])
    campsites = MagicMock(return_value=[])
    monkeypatch.setattr("source.agent.search.search_open_slots", slots)
    monkeypatch.setattr("source.agent.search.search_stated_amenities", amenities)
    monkeypatch.setattr("source.agent.search.search_review_claims", claims)
    monkeypatch.setattr("source.agent.search.lookup_campsite_by_name", lookup)
    monkeypatch.setattr("source.agent.search.search_campsites", campsites)
    return SimpleNamespace(
        slots=slots,
        amenities=amenities,
        claims=claims,
        lookup=lookup,
        campsites=campsites,
    )


def test_catalog_date_party_ac_intersects_on_type_ids(two_stage: SimpleNamespace):
    result = planner_node(
        _constraints_state(
            {
                "date": DATE,
                "numeric_constraints": [
                    {"field": "party_size", "operator": "=", "value": 3}
                ],
                "semantic_constraints": [{"query": "air conditioning"}],
            }
        )
    )
    two_stage.lookup.assert_not_called()
    two_stage.slots.assert_called_once()
    kwargs = two_stage.slots.call_args.kwargs
    assert kwargs["date_range"] == DATE
    assert kwargs["site_id"] is None
    assert kwargs["party_size"] == 3
    two_stage.amenities.assert_called_once()
    amenity_kwargs = two_stage.amenities.call_args.kwargs
    assert amenity_kwargs["accommodation_type_ids"] == [11, 22]
    assert amenity_kwargs["embedding"] == FAKE_VEC

    payload = _fits_payload(result)
    assert payload["rejected_count"] == 1
    assert len(payload["fits"]) == 1
    assert len(payload["rejected"]) == 1
    miss = payload["rejected"][0]
    assert miss["accommodation_type_id"] == 22
    assert miss["why"] == [
        {"reason": "missing_stated_amenity", "query": "air conditioning"}
    ]
    fit = payload["fits"][0]
    assert fit["accommodation_type_id"] == 11
    assert fit["price_per_night"] == 400.0
    assert fit["why"] == [
        {
            "query": "air conditioning",
            "stated_amenity": "air_conditioning",
            "distance": -0.9,
        }
    ]
    two_stage.campsites.assert_not_called()


def test_named_park_passes_site_id(two_stage: SimpleNamespace):
    two_stage.lookup.return_value = [
        {"id": 1, "name": "חורשת טל", "hotel_id": 1, "booking_hotel_id": "9_1"}
    ]
    two_stage.slots.return_value = [dict(SLOT_AC, campsite_id=1)]
    planner_node(
        _constraints_state(
            {
                "date": DATE,
                "campsite": "Horshat Tal",
                "numeric_constraints": [
                    {"field": "party_size", "operator": "=", "value": 3}
                ],
                "semantic_constraints": [{"query": "air conditioning"}],
            }
        )
    )
    two_stage.lookup.assert_called_once_with("Horshat Tal")
    assert two_stage.slots.call_args.kwargs["site_id"] == 1


def test_planner_passes_price_constraint_to_slots(two_stage: SimpleNamespace):
    two_stage.slots.return_value = [dict(SLOT_TENT)]
    two_stage.amenities.return_value = []
    planner_node(
        _constraints_state(
            {
                "date": DATE,
                "numeric_constraints": [
                    {"field": "price_per_night", "operator": "<=", "value": 200}
                ],
                "semantic_constraints": [],
            }
        )
    )
    numeric = two_stage.slots.call_args.kwargs["numeric_constraints"]
    assert numeric == [{"field": "price_per_night", "operator": "<=", "value": 200}]
    two_stage.amenities.assert_not_called()


def test_no_date_skips_vacancy_and_semantic(two_stage: SimpleNamespace):
    result = planner_node(
        _constraints_state(
            {
                "date": None,
                "numeric_constraints": [
                    {"field": "party_size", "operator": "=", "value": 3}
                ],
                "semantic_constraints": [{"query": "air conditioning"}],
            }
        )
    )
    two_stage.slots.assert_not_called()
    two_stage.amenities.assert_not_called()
    two_stage.lookup.assert_not_called()
    payload = _fits_payload(result)
    assert payload["fits"] == []
    assert payload["skipped"] == "no_date"


def test_price_constraint_helpers():
    assert _price_per_night_constraint(
        [{"field": "price_per_night", "operator": "<=", "value": 500}]
    ) == ("<=", 500.0)
    assert _price_matches(400.0, ("<=", 500.0)) is True
    assert _price_matches(600.0, ("<=", 500.0)) is False
    assert _price_matches(None, ("<=", 500.0)) is False
    assert _price_matches(None, None) is True


def test_rate_period_weekend_in_range():
    assert _rate_period_for_stay(DATE) == "weekend_holiday"
    assert (
        _rate_period_for_stay({"start": "2026-08-31", "end": "2026-09-01"})
        == "weekday"
    )
