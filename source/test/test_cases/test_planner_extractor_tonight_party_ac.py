"""Planner output for the extractor JSON of: 3 people, 2 nights from today, AC."""

from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Self
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage

from source.agent import search as agent_search
from source.agent.graph import (
    _open_slots_sql,
    _render_sql,
    planner_node,
    search_open_slots,
)

EXTRACTOR_JSON = {
    "date": {"start": "2026-08-30", "end": "2026-09-01"},
    "campsite": None,
    "numeric_constraints": [
        {"field": "party_size", "operator": "=", "value": 3}
    ],
    "semantic_constraints": [{"query": "air conditioning"}],
}

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


def _planner_payload(result: dict) -> dict:
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, ChatMessage)
    data = json.loads(str(msg.content))
    assert isinstance(data, dict)
    return data


@pytest.fixture
def two_stage(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        "source.agent.search._query_vec_literal", lambda query: "[0]"
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
    monkeypatch.setattr("source.agent.search.search_open_slots", slots)
    monkeypatch.setattr("source.agent.search.search_stated_amenities", amenities)
    monkeypatch.setattr(
        "source.agent.search.search_review_claims", MagicMock(return_value=[])
    )
    monkeypatch.setattr(
        "source.agent.search.lookup_campsite_by_name", MagicMock(return_value=[])
    )
    return SimpleNamespace(slots=slots, amenities=amenities)


def test_planner_output_for_extractor_tonight_party_ac(two_stage: SimpleNamespace):
    result = planner_node(
        {
            "messages": [
                HumanMessage(
                    content="משהו ל 3 מבוגרים ל 2 לילות החל מהיום, עם מזגן"
                ),
                AIMessage(content=json.dumps(EXTRACTOR_JSON, ensure_ascii=False)),
            ]
        }
    )
    two_stage.slots.assert_called_once_with(
        date_range={"start": "2026-08-30", "end": "2026-09-01"},
        site_id=None,
        party_size=3,
        numeric_constraints=EXTRACTOR_JSON["numeric_constraints"],
    )
    two_stage.amenities.assert_called_once()
    assert two_stage.amenities.call_args.args[0] == "air conditioning"
    assert two_stage.amenities.call_args.kwargs["accommodation_type_ids"] == [
        11,
        22,
    ]

    payload = _planner_payload(result)
    assert payload["constraints"]["date"] == EXTRACTOR_JSON["date"]
    assert (
        payload["constraints"]["numeric_constraints"]
        == EXTRACTOR_JSON["numeric_constraints"]
    )
    assert payload["constraints"]["semantic_constraints"] == [
        {"query": "air conditioning", "locus": "site"}
    ], "a locus-less extractor item defaults to site"
    assert "campsite" not in payload["constraints"]
    assert payload["rejected_count"] == 1
    assert [fit["accommodation_type_id"] for fit in payload["fits"]] == [11]
    assert payload["fits"][0]["why"] == [
        {
            "query": "air conditioning",
            "stated_amenity": "air_conditioning",
            "distance": -0.9,
        }
    ]
    assert payload["rejected"][0]["accommodation_type_id"] == 22
    assert payload["rejected"][0]["why"] == [
        {"reason": "missing_stated_amenity", "query": "air conditioning"}
    ]


def test_open_slots_sql_for_extractor_tonight_party_ac():
    sql, params = _open_slots_sql(
        date_range=EXTRACTOR_JSON["date"],
        site_id=None,
        party_size=3,
        limit=80,
    )
    rendered = _render_sql(sql, params)
    assert "a.start_date >= '2026-08-30'" in rendered
    assert "a.start_date < '2026-09-01'" in rendered
    assert "a.end_date = a.start_date + 1" in rendered
    assert "HAVING COUNT(DISTINCT a.start_date) = 2" in rendered
    assert "(at.max_occupancy IS NULL OR at.max_occupancy >= 3)" in rendered
    assert "LIMIT 80" in rendered
    assert "%s" not in rendered
    where = rendered.split("WHERE", 1)[1].split("GROUP BY", 1)[0]
    assert "a.site_id" not in where


# One-night availability rows the mock DB holds for 2026-08-30 → 2026-09-01.
# Columns match search_open_slots: site_id, campsite, start, end, rooms, type_id, type, occ.
_MOCK_NIGHTS = [
    (3, "Park A", date(2026, 8, 30), date(2026, 8, 31), 2, 11, "בונגלו עם מזגן", 4),
    (3, "Park A", date(2026, 8, 31), date(2026, 9, 1), 1, 11, "בונגלו עם מזגן", 4),
    (4, "Park B", date(2026, 8, 30), date(2026, 8, 31), 3, 22, "אוהל", 6),
]


def _mock_availability_result(sql: str, params: list) -> list[tuple]:
    stay_start, stay_end = params[0], params[1]
    night_count = params[-2]
    limit = params[-1]
    party_size = params[-3] if "max_occupancy" in sql else None
    kept: list[tuple] = []
    for row in _MOCK_NIGHTS:
        start, end, occ = row[2], row[3], row[7]
        if start < stay_start or start >= stay_end:
            continue
        if end != start + timedelta(days=1):
            continue
        if party_size is not None and occ is not None and occ < party_size:
            continue
        kept.append(row)
    grouped: dict[tuple, list[tuple]] = {}
    for row in kept:
        grouped.setdefault((row[0], row[1], row[5], row[6], row[7]), []).append(row)
    out: list[tuple] = []
    for key, rows in grouped.items():
        if len({row[2] for row in rows}) != night_count:
            continue
        out.append(
            (
                key[0],
                key[1],
                min(row[2] for row in rows),
                max(row[3] for row in rows),
                min(row[4] for row in rows),
                key[2],
                key[3],
                key[4],
            )
        )
    out.sort(key=lambda row: (row[2], row[5]))
    return out[:limit]


class _MockCursor:
    def __init__(self) -> None:
        self._rows: list[tuple] = []

    def execute(self, sql: str, params=None) -> None:
        params = list(params or [])
        if "FROM availability" in sql:
            self._rows = _mock_availability_result(sql, params)
        else:
            self._rows = []

    def fetchall(self) -> list[tuple]:
        return self._rows

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        return None


class _MockConn:
    def cursor(self) -> _MockCursor:
        return _MockCursor()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        return None


def test_open_slots_queries_db_for_extractor_tonight_party_ac(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DATABASE_URL", "postgresql://mock")
    monkeypatch.setattr(agent_search.psycopg, "connect", lambda _url: _MockConn())

    slots = search_open_slots(
        date_range=EXTRACTOR_JSON["date"],
        site_id=None,
        party_size=3,
        numeric_constraints=EXTRACTOR_JSON["numeric_constraints"],
    )
    recorded = agent_search._LAST_OPEN_SLOTS_QUERY
    assert recorded is not None
    assert recorded.get("skipped") is None
    if slots and slots[0].get("error"):
        pytest.fail(slots[0]["error"])
    assert recorded.get("row_count") == 1
    assert [slot["accommodation_type_id"] for slot in slots] == [11]
    assert slots[0]["campsite"] == "Park A"
    assert slots[0]["start"] == "2026-08-30"
    assert slots[0]["end"] == "2026-09-01"
    assert slots[0]["room_count"] == 1
    assert slots[0]["max_occupancy"] == 4
