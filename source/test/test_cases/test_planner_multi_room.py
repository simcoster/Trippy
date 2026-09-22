"""Vacancy search still requires one unit to sleep the party.

Multi-room stays are not implemented. Open slots keep
`(max_occupancy IS NULL OR max_occupancy >= party_size)`, and the planner
does not combine a bungalow and a tent into one stay.

`room_count` on a slot is inventory (how many of that type are free).
The planner does not set `units` (how many of that type to book).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage

from source.agent.graph import _open_slots_sql, _render_sql, planner_node

DATE = {"start": "2026-08-30", "end": "2026-08-31"}
PARTY_SIX = [{"field": "party_size", "operator": ">=", "value": 6}]

SLOT_BUNGALOW = {
    "campsite_id": 3,
    "campsite": "Park A",
    "start": "2026-08-30",
    "end": "2026-08-31",
    "room_count": 3,
    "accommodation_type_id": 11,
    "accommodation_type": "בונגלו",
    "max_occupancy": 4,
    "occupancy_unknown": False,
    "price_per_night": 400.0,
}
SLOT_TENT = {
    "campsite_id": 3,
    "campsite": "Park A",
    "start": "2026-08-30",
    "end": "2026-08-31",
    "room_count": 2,
    "accommodation_type_id": 22,
    "accommodation_type": "אוהל",
    "max_occupancy": 2,
    "occupancy_unknown": False,
    "price_per_night": 180.0,
}


def _constraints_state(constraints: dict) -> dict:
    return {
        "messages": [
            HumanMessage(content="6 people, one night, two rooms if needed"),
            AIMessage(content=json.dumps(constraints, ensure_ascii=False)),
        ]
    }


def _fits_payload(result: dict) -> dict:
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, ChatMessage)
    data = json.loads(str(msg.content))
    assert isinstance(data, dict)
    return data


def _fit_units(fit: dict) -> list[dict]:
    rooms = fit.get("rooms")
    if isinstance(rooms, list) and rooms:
        return rooms
    if fit.get("units") is not None or fit.get("accommodation_type_id") is not None:
        return [
            {
                "accommodation_type_id": fit.get("accommodation_type_id"),
                "units": fit.get("units"),
                "max_occupancy": fit.get("max_occupancy"),
            }
        ]
    return []


def _capacity(fit: dict) -> int:
    if fit.get("party_capacity") is not None:
        return int(fit["party_capacity"])
    total = 0
    for room in _fit_units(fit):
        occ = room.get("max_occupancy")
        units = room.get("units")
        if occ is None or units is None:
            continue
        total += int(occ) * int(units)
    return total


@pytest.fixture
def vacancy_search(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        "source.agent.search.embed._query_vec_literal", lambda query: "[0]"
    )
    slots = MagicMock()
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


def test_open_slots_sql_requires_one_unit_to_sleep_the_party():
    """Multi-room is not implemented: one unit's occupancy must cover 6."""
    sql, params = _open_slots_sql(
        windows=[DATE],
        site_id=None,
        party_size=6,
        limit=80,
    )
    assert sql is not None
    rendered = _render_sql(sql, params)
    assert "(at.max_occupancy IS NULL OR at.max_occupancy >= 6)" in rendered, (
        rendered
    )


def test_planner_books_two_units_of_same_type_for_party_of_six(
    vacancy_search: SimpleNamespace,
):
    """One 4-person bungalow is too small; two of three available units fit 6."""
    vacancy_search.slots.return_value = [dict(SLOT_BUNGALOW)]
    result = planner_node(
        _constraints_state(
            {
                "date": DATE,
                "numeric_constraints": PARTY_SIX,
                "semantic_constraints": [],
            }
        )
    )
    vacancy_search.slots.assert_called_once_with(
        date_windows=[DATE],
        site_id=None,
        party_size=6,
        numeric_constraints=PARTY_SIX,
    )
    payload = _fits_payload(result)
    assert len(payload["fits"]) == 1, payload["fits"]
    fit = payload["fits"][0]
    assert fit["campsite_id"] == 3
    rooms = _fit_units(fit)
    assert len(rooms) == 1, rooms
    assert rooms[0]["accommodation_type_id"] == 11
    assert rooms[0]["units"] == 2
    assert _capacity(fit) >= 6
    assert fit.get("room_count") == 3


def test_planner_keeps_mixed_types_as_separate_fits(vacancy_search: SimpleNamespace):
    """A bungalow and a tent at one park stay two fits. Multi-room is not implemented."""
    vacancy_search.slots.return_value = [dict(SLOT_BUNGALOW), dict(SLOT_TENT)]
    result = planner_node(
        _constraints_state(
            {
                "date": DATE,
                "numeric_constraints": PARTY_SIX,
                "semantic_constraints": [],
            }
        )
    )
    payload = _fits_payload(result)
    fits = [fit for fit in payload["fits"] if fit.get("campsite_id") == 3]
    assert len(fits) == 2, payload["fits"]
    by_type = {int(fit["accommodation_type_id"]): fit for fit in fits}
    assert set(by_type) == {11, 22}
    for fit in fits:
        assert fit.get("units") is None
        assert not fit.get("rooms")
        assert len(_fit_units(fit)) == 1
        assert _capacity(fit) < 6
