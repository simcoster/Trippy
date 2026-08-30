"""Planner dispatch: which searches run for date / semantic / numeric constraints.

DB-facing helpers are mocked — these tests check query selection, not SQL or RAG quality.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage

from source.agent.graph import planner_node

FAKE_VEC = "[0.1,0.2,0.3]"


def _constraints_state(constraints: dict) -> dict:
    return {
        "messages": [
            HumanMessage(content="plan a trip"),
            AIMessage(content=json.dumps(constraints, ensure_ascii=False)),
        ]
    }


def _chat_texts(result: dict) -> list[str]:
    return [
        str(msg.content)
        for msg in result["messages"]
        if isinstance(msg, ChatMessage)
    ]


def _payloads(result: dict) -> list[dict]:
    out: list[dict] = []
    for text in _chat_texts(result):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "stated_amenities" in data:
            out.append(data)
    return out


@pytest.fixture
def db_searches(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        "source.agent.graph._query_vec_literal", lambda query: FAKE_VEC
    )
    amenities = MagicMock(return_value=[])
    claims = MagicMock(return_value=[])
    campsites = MagicMock(return_value=[])
    monkeypatch.setattr("source.agent.graph.search_stated_amenities", amenities)
    monkeypatch.setattr("source.agent.graph.search_review_claims", claims)
    monkeypatch.setattr("source.agent.graph.search_campsites", campsites)
    return SimpleNamespace(
        amenities=amenities, claims=claims, campsites=campsites
    )


# ---- dates (no availability SQL yet — note only, no DB) ----


def test_planner_date_range_emits_stay_nights_without_db(db_searches: SimpleNamespace):
    result = planner_node(
        _constraints_state(
            {
                "date": {"start": "2026-08-30", "end": "2026-09-01"},
                "numeric_constraints": [],
                "semantic_constraints": [],
            }
        )
    )
    assert _chat_texts(result) == [
        "Requested stay nights: 2026-08-30 .. 2026-09-01"
    ]
    db_searches.amenities.assert_not_called()
    db_searches.claims.assert_not_called()
    db_searches.campsites.assert_not_called()


def test_planner_missing_date_skips_stay_nights(db_searches: SimpleNamespace):
    result = planner_node(
        _constraints_state(
            {
                "date": None,
                "numeric_constraints": [],
                "semantic_constraints": [{"query": "quiet"}],
            }
        )
    )
    assert not any("Requested stay nights" in text for text in _chat_texts(result))
    db_searches.campsites.assert_not_called()


# ---- semantic_constraints → amenity RAG + claims RAG ----


def test_planner_semantic_query_hits_amenities_and_claims(
    db_searches: SimpleNamespace,
):
    db_searches.amenities.return_value = [
        {"amenity": "air_conditioning", "hotel_id": 3}
    ]
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
                "date": None,
                "numeric_constraints": [],
                "semantic_constraints": [{"query": "air conditioning"}],
            }
        )
    )
    db_searches.amenities.assert_called_once_with(
        "air conditioning", limit=5, embedding=FAKE_VEC
    )
    db_searches.claims.assert_called_once_with(
        "air conditioning", limit=5, embedding=FAKE_VEC
    )
    db_searches.campsites.assert_not_called()

    payload = _payloads(result)[0]
    assert payload["query"] == "air conditioning"
    assert payload["stated_amenities"] == ["air_conditioning"]
    assert payload["review_claims"] == [
        {
            "claim": "the AC was loud",
            "date": "2026-05-01",
            "days_ago": 121,
            "campsite_id": "3",
        }
    ]


def test_planner_semantic_and_runs_each_query(db_searches: SimpleNamespace):
    planner_node(
        _constraints_state(
            {
                "date": None,
                "numeric_constraints": [],
                "semantic_constraints": [
                    {"query": "air conditioning"},
                    {"query": "quiet"},
                ],
            }
        )
    )
    amenity_queries = [
        c.kwargs.get("query") or (c.args[0] if c.args else None)
        for c in db_searches.amenities.call_args_list
    ]
    claim_queries = [
        c.kwargs.get("query") or (c.args[0] if c.args else None)
        for c in db_searches.claims.call_args_list
    ]
    assert amenity_queries == ["air conditioning", "quiet"]
    assert claim_queries == ["air conditioning", "quiet"]
    for c in db_searches.amenities.call_args_list:
        assert c.kwargs["limit"] == 5
        assert c.kwargs["embedding"] == FAKE_VEC


def test_planner_semantic_or_group_searches_both_values(db_searches: SimpleNamespace):
    result = planner_node(
        _constraints_state(
            {
                "date": None,
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
    amenity_queries = [
        c.args[0] for c in db_searches.amenities.call_args_list
    ]
    claim_queries = [c.args[0] for c in db_searches.claims.call_args_list]
    assert amenity_queries == ["near the sea", "near a body of water"]
    assert claim_queries == ["near the sea", "near a body of water"]
    # One merged payload for the OR group (not two AND messages).
    payloads = _payloads(result)
    assert len(payloads) == 1
    assert payloads[0]["query"] == ["near the sea", "near a body of water"]
    db_searches.campsites.assert_not_called()


# ---- numeric_constraints: never dump the campsite catalog ----


def test_planner_party_size_does_not_dump_campsite_catalog(
    db_searches: SimpleNamespace,
):
    numeric = [{"field": "party_size", "operator": "=", "value": 3}]
    result = planner_node(
        _constraints_state(
            {
                "date": None,
                "numeric_constraints": numeric,
                "semantic_constraints": [],
            }
        )
    )
    db_searches.campsites.assert_not_called()
    db_searches.amenities.assert_not_called()
    db_searches.claims.assert_not_called()
    texts = _chat_texts(result)
    assert any("party_size" in text for text in texts)
    assert not any("parks.org.il" in text for text in texts)


def test_planner_price_constraint_does_not_dump_campsite_catalog(
    db_searches: SimpleNamespace,
):
    numeric = [{"field": "price_per_night", "operator": "<=", "value": 500}]
    planner_node(
        _constraints_state(
            {
                "date": None,
                "numeric_constraints": numeric,
                "semantic_constraints": [],
            }
        )
    )
    db_searches.campsites.assert_not_called()


def test_planner_empty_numeric_skips_campsites(db_searches: SimpleNamespace):
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
    db_searches.amenities.assert_called_once()
    db_searches.claims.assert_called_once()
