"""The planner runs the sandbox quote and retrieve together after vacancies."""

from __future__ import annotations

import threading

from source.agent.planner import _SemanticWhy, planner_fits_payload

_SLOT = {
    "campsite_id": 1,
    "campsite": "Park",
    "start": "2026-09-04",
    "end": "2026-09-05",
    "room_count": 1,
    "accommodation_type_id": 2,
    "accommodation_type": "tent",
    "max_occupancy": 4,
    "occupancy_unknown": False,
}


def test_planner_quotes_while_retrieving(monkeypatch):
    quote_started = threading.Event()
    retrieve_started = threading.Event()

    def search(**_kwargs):
        return [dict(_SLOT)]

    def quote(slots, **_kwargs):
        quote_started.set()
        assert retrieve_started.wait(2), "retrieve did not start during the quote"
        priced = [dict(slot) for slot in slots]
        priced[0]["price_per_night"] = 10.0
        return priced

    def semantic(_slots, _constraints):
        assert quote_started.wait(2), "quote did not start before retrieve"
        retrieve_started.set()
        return _SemanticWhy(
            why_by_slot={},
            reject_why_by_slot={},
            claims_by_site={},
            rules_by_query={},
        )

    monkeypatch.setattr("source.agent.search.availability.search_open_slots", search)
    monkeypatch.setattr("source.agent.search.availability.quote_open_slots", quote)
    monkeypatch.setattr("source.agent.planner._semantic_why_by_slot", semantic)
    payload = planner_fits_payload(
        {
            "date": {"start": "2026-09-04", "end": "2026-09-05"},
            "numeric_constraints": [],
            "semantic_constraints": [],
        }
    )
    assert payload["fits"] == []
    assert payload["rejected"][0]["price_per_night"] == 10.0
