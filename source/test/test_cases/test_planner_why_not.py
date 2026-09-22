"""Price misses stay on the rejected list, and why_not counts campsites."""

from source.agent.planner import planner_fits_payload
from source.agent.search.availability import (
    _GatheredQuotes,
    quote_open_slots,
)
from source.agent.search.sandbox import _SandboxQuoteBatch

SLOT = {
    "campsite_id": 1,
    "campsite": "Park A",
    "accommodation_type_id": 11,
    "accommodation_type": "tent",
    "start": "2026-05-24",
    "end": "2026-05-25",
    "room_count": 1,
    "max_occupancy": 4,
    "occupancy_unknown": False,
    "price_per_night": 180.0,
}


def test_quote_records_slots_outside_the_price_range(monkeypatch):
    import source.agent.search.availability as availability

    monkeypatch.setattr(
        availability,
        "_gather_slot_quotes",
        lambda *args, **kwargs: _GatheredQuotes(
            prices={},
            batch=_SandboxQuoteBatch(by_key={}, report={"calls": []}),
        ),
    )
    availability._LAST_OPEN_SLOTS_QUERY = {}
    quoted = quote_open_slots(
        [dict(SLOT)],
        party_size=2,
        numeric_constraints=[
            {"field": "price_per_night", "operator": "<=", "value": 200}
        ],
    )
    assert quoted == []
    missed = availability._LAST_OPEN_SLOTS_QUERY["price_rejected"]
    assert missed[0]["campsite_id"] == 1
    assert missed[0]["price_per_night"] is None


def test_price_and_amenity_funnel_counts_campsites(monkeypatch):
    import source.agent.search.availability as availability

    pool = dict(SLOT, campsite_id=1, campsite="Pool", accommodation_type_id=11)
    dry = dict(
        SLOT,
        campsite_id=2,
        campsite="Dry",
        accommodation_type_id=22,
        accommodation_type="cabin",
    )
    pricey_a = dict(
        SLOT,
        campsite_id=3,
        campsite="Pricey",
        accommodation_type_id=33,
        accommodation_type="suite",
        price_per_night=900.0,
    )
    pricey_b = dict(
        SLOT,
        campsite_id=3,
        campsite="Pricey",
        accommodation_type_id=34,
        accommodation_type="villa",
        price_per_night=950.0,
    )
    availability._LAST_OPEN_SLOTS_QUERY = {}
    monkeypatch.setattr(
        availability,
        "search_open_slots",
        lambda **kwargs: [pool, dry, pricey_a, pricey_b],
    )

    def _quote(slots, **kwargs):
        return [slot for slot in slots if slot["campsite_id"] != 3]

    monkeypatch.setattr(availability, "quote_open_slots", _quote)
    monkeypatch.setattr(
        "source.agent.search.embed._query_vec_literals",
        lambda queries: {query: "[0]" for query in queries},
    )
    monkeypatch.setattr(
        "source.agent.search.amenities.search_stated_amenities",
        lambda *args, **kwargs: [
            {
                "amenity": "pool",
                "accommodation_type_id": 11,
                "distance": -0.9,
            }
        ],
    )
    monkeypatch.setattr(
        "source.agent.search.amenities.search_site_amenities",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "source.agent.search.claims.search_review_claims",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "source.agent.search.rules.search_campsite_rules",
        lambda *args, **kwargs: [],
    )

    payload = planner_fits_payload(
        {
            "date": {"start": "2026-05-24", "end": "2026-05-25"},
            "numeric_constraints": [
                {"field": "price_per_night", "operator": "<=", "value": 400}
            ],
            "semantic_constraints": [{"query": "pool"}],
        }
    )

    assert [fit["campsite_id"] for fit in payload["fits"]] == [1]
    price_rows = [
        row
        for row in payload["rejected"]
        if row["why"] == [{"reason": "price"}]
    ]
    assert {row["accommodation_type_id"] for row in price_rows} == {33, 34}
    assert payload["rejected_count"] == 3
    assert payload["why_not"] == [
        {"stage": "price", "count": 1, "sites": ["Pricey"]},
        {
            "stage": "missing",
            "count": 1,
            "query": "pool",
            "sites": ["Dry"],
        },
    ]
    assert "price_rejected" not in payload["open_slots_query"]
