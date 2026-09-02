"""Planner / vacancy search: compose multiple rooms so occupancy covers the party.

Currently fails: stage 1 requires one accommodation type to sleep everyone
(`at.max_occupancy >= party_size`). A 6-person party cannot take two 4-person
bungalows even when `availability.room_count >= 2`, and cannot mix a bungalow
plus a tent at the same site.

`room_count` on a slot is inventory (how many of that type are free).
`units` is how many of that type the party should book.
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
        "source.agent.search._query_vec_literal", lambda query: "[0]"
    )
    slots = MagicMock()
    monkeypatch.setattr("source.agent.search.search_open_slots", slots)
    monkeypatch.setattr(
        "source.agent.search.search_stated_amenities", MagicMock(return_value=[])
    )
    monkeypatch.setattr(
        "source.agent.search.search_review_claims", MagicMock(return_value=[])
    )
    monkeypatch.setattr(
        "source.agent.search.lookup_campsite_by_name", MagicMock(return_value=[])
    )
    return SimpleNamespace(slots=slots)


def test_open_slots_sql_does_not_require_one_unit_to_sleep_the_party():
    """6 guests can take two 4-person bungalows; do not require occupancy >= 6."""
    sql, params = _open_slots_sql(
        date_range=DATE,
        site_id=None,
        party_size=6,
        limit=80,
    )
    assert sql is not None
    rendered = _render_sql(sql, params)
    assert "at.max_occupancy >= 6" not in rendered.replace(" ", ""), rendered
    assert "(at.max_occupancy IS NULL OR at.max_occupancy >= 6)" not in rendered, (
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
        date_range=DATE,
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


def test_planner_composes_mixed_types_at_one_site(vacancy_search: SimpleNamespace):
    """Bungalow (4) + tent (2) at the same park cover a party of 6."""
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
    composed = [
        fit
        for fit in payload["fits"]
        if fit.get("campsite_id") == 3 and _capacity(fit) >= 6
    ]
    assert len(composed) == 1, payload["fits"]
    fit = composed[0]
    type_ids = {
        int(room["accommodation_type_id"])
        for room in _fit_units(fit)
        if room.get("accommodation_type_id") is not None
    }
    assert type_ids == {11, 22}, _fit_units(fit)
    by_type = {
        int(room["accommodation_type_id"]): int(room["units"])
        for room in _fit_units(fit)
    }
    assert by_type[11] == 1
    assert by_type[22] == 1
