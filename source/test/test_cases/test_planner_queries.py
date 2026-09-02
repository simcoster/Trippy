"""Planner dispatch: vacancies first, then in-set semantic intersection.

DB-facing helpers are mocked — these tests check query selection, not SQL.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage

from source.agent.graph import planner_node

FAKE_VEC = "[0.1,0.2,0.3]"
DATE = {"start": "2026-08-30", "end": "2026-09-01"}

SLOT = {
    "campsite_id": 3,
    "campsite": "Park A",
    "start": "2026-08-30",
    "end": "2026-09-01",
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
    for msg in result["messages"]:
        if not isinstance(msg, ChatMessage):
            continue
        data = json.loads(str(msg.content))
        if isinstance(data, dict) and "fits" in data:
            return data
    raise AssertionError("planner did not return a fits payload")


@pytest.fixture
def db_searches(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        "source.agent.search._query_vec_literal", lambda query: FAKE_VEC
    )
    slots = MagicMock(return_value=[dict(SLOT)])
    amenities = MagicMock(
        return_value=[
            {
                "amenity": "air_conditioning",
                "accommodation_type_id": 11,
                "distance": -0.9,
            }
        ]
    )
    claims = MagicMock(return_value=[])
    campsites = MagicMock(return_value=[])
    monkeypatch.setattr("source.agent.search.search_open_slots", slots)
    monkeypatch.setattr("source.agent.search.search_stated_amenities", amenities)
    monkeypatch.setattr("source.agent.search.search_review_claims", claims)
    monkeypatch.setattr("source.agent.search.search_campsites", campsites)
    return SimpleNamespace(
        slots=slots, amenities=amenities, claims=claims, campsites=campsites
    )


def test_planner_date_range_queries_open_slots(db_searches: SimpleNamespace):
    result = planner_node(
        _constraints_state(
            {
                "date": DATE,
                "numeric_constraints": [],
                "semantic_constraints": [],
            }
        )
    )
    db_searches.slots.assert_called_once()
    assert db_searches.slots.call_args.kwargs["date_range"] == DATE
    assert db_searches.slots.call_args.kwargs["site_id"] is None
    db_searches.amenities.assert_not_called()
    db_searches.claims.assert_not_called()
    db_searches.campsites.assert_not_called()
    payload = _fits_payload(result)
    assert len(payload["fits"]) == 1
    assert payload["fits"][0]["accommodation_type_id"] == 11
    assert payload["fits"][0]["why"] == []


def test_planner_missing_date_skips_vacancy(db_searches: SimpleNamespace):
    result = planner_node(
        _constraints_state(
            {
                "date": None,
                "numeric_constraints": [],
                "semantic_constraints": [{"query": "quiet"}],
            }
        )
    )
    db_searches.slots.assert_not_called()
    db_searches.amenities.assert_not_called()
    db_searches.campsites.assert_not_called()
    payload = _fits_payload(result)
    assert payload["fits"] == []
    assert payload["skipped"] == "no_date"


def test_planner_semantic_query_restricted_to_slot_types(
    db_searches: SimpleNamespace,
):
    db_searches.claims.return_value = [
        {
            "claim": "the AC was loud",
            "date": "2026-05-01",
            "days_ago": 121,
            "campsite_id": "3",
        }
    ]
    result = planner_node(
        _constraints_state(
            {
                "date": DATE,
                "numeric_constraints": [],
                "semantic_constraints": [{"query": "air conditioning"}],
            }
        )
    )
    db_searches.amenities.assert_called_once_with(
        "air conditioning",
        limit=1,
        embedding=FAKE_VEC,
        accommodation_type_ids=[11],
    )
    db_searches.claims.assert_called_once_with(
        "air conditioning", limit=5, embedding=FAKE_VEC, campsite_ids=[3]
    )
    db_searches.campsites.assert_not_called()

    fit = _fits_payload(result)["fits"][0]
    assert fit["why"][0]["stated_amenity"] == "air_conditioning"
    assert fit["review_claims"] == [
        {
            "query": "air conditioning",
            "claim": "the AC was loud",
            "date": "2026-05-01",
            "days_ago": 121,
            "is_positive": None,
        }
    ]


def test_planner_semantic_and_runs_each_query(db_searches: SimpleNamespace):
    planner_node(
        _constraints_state(
            {
                "date": DATE,
                "numeric_constraints": [],
                "semantic_constraints": [
                    {"query": "air conditioning"},
                    {"query": "quiet"},
                ],
            }
        )
    )
    amenity_queries = [
        c.args[0] for c in db_searches.amenities.call_args_list
    ]
    assert amenity_queries == ["air conditioning", "quiet"]
    for c in db_searches.amenities.call_args_list:
        assert c.kwargs["accommodation_type_ids"] == [11]
        assert c.kwargs["embedding"] == FAKE_VEC


def test_planner_semantic_or_group_searches_both_values(db_searches: SimpleNamespace):
    result = planner_node(
        _constraints_state(
            {
                "date": DATE,
                "numeric_constraints": [],
                "semantic_constraints": [
                    {
                        "op": "or",
                        "values": ["near the sea", "near a body of water"],
                    }
                ],
            }
        )
    )
    amenity_queries = [c.args[0] for c in db_searches.amenities.call_args_list]
    assert amenity_queries == ["near the sea", "near a body of water"]
    payload = _fits_payload(result)
    assert len(payload["fits"]) == 1
    db_searches.campsites.assert_not_called()


def test_planner_party_size_goes_to_open_slots_not_catalog(
    db_searches: SimpleNamespace,
):
    numeric = [{"field": "party_size", "operator": "=", "value": 3}]
    result = planner_node(
        _constraints_state(
            {
                "date": DATE,
                "numeric_constraints": numeric,
                "semantic_constraints": [],
            }
        )
    )
    db_searches.campsites.assert_not_called()
    db_searches.amenities.assert_not_called()
    assert db_searches.slots.call_args.kwargs["party_size"] == 3
    texts = json.dumps(_fits_payload(result), ensure_ascii=False)
    assert "parks.org.il" not in texts


def test_planner_price_constraint_goes_to_open_slots(
    db_searches: SimpleNamespace,
):
    numeric = [{"field": "price_per_night", "operator": "<=", "value": 500}]
    planner_node(
        _constraints_state(
            {
                "date": DATE,
                "numeric_constraints": numeric,
                "semantic_constraints": [],
            }
        )
    )
    db_searches.campsites.assert_not_called()
    assert db_searches.slots.call_args.kwargs["numeric_constraints"] == numeric


def test_planner_empty_numeric_still_queries_slots(db_searches: SimpleNamespace):
    planner_node(
        _constraints_state(
            {
                "date": {"start": "2026-08-28", "end": "2026-08-29"},
                "numeric_constraints": [],
                "semantic_constraints": [{"query": "hot showers"}],
            }
        )
    )
    db_searches.campsites.assert_not_called()
    db_searches.slots.assert_called_once()
    db_searches.amenities.assert_called_once()
    db_searches.claims.assert_called_once()
