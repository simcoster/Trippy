"""Room vs. site amenity locus, and claims as a match / evidence / ranking lane.

"מקום עם מקררים" ("a place with fridges") is a site ask — a campsite with a
communal fridge block satisfies it. "יש מקרר בחדר" ("there's a fridge in the
room") is a room ask — communal fridges are not enough, the booked unit must
list one.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage

from source.agent.constraints import (
    normalize_constraints,
    normalize_locus,
    semantic_locus_groups,
)
from source.agent.graph import planner_node
from source.agent.planner import CLAIM_RECENCY_HALF_LIFE_DAYS

FAKE_VEC = "[0.1,0.2,0.3]"

DATE = {"start": "2026-09-04", "end": "2026-09-05"}

# Campsite 3 has a communal fridge block; its tent lists no fridge.
SLOT_COMMUNAL = {
    "campsite_id": 3,
    "campsite": "Park Communal",
    "start": "2026-09-04",
    "end": "2026-09-05",
    "room_count": 1,
    "accommodation_type_id": 11,
    "accommodation_type": "אוהל",
    "max_occupancy": 4,
    "occupancy_unknown": False,
    "price_per_night": 180.0,
}
# Campsite 4's bungalow lists a fridge of its own.
SLOT_IN_ROOM = {
    "campsite_id": 4,
    "campsite": "Park Cabins",
    "start": "2026-09-04",
    "end": "2026-09-05",
    "room_count": 1,
    "accommodation_type_id": 22,
    "accommodation_type": "בונגלו",
    "max_occupancy": 4,
    "occupancy_unknown": False,
    "price_per_night": 400.0,
}

TYPE_FRIDGE_HIT = {
    "amenity": "refrigerator",
    "accommodation_type_id": 22,
    "accommodation_type": SLOT_IN_ROOM["accommodation_type"],
    "hotel_id": 4,
    "distance": -0.9,
}
SITE_FRIDGE_HIT = {
    "amenity": "communal_refrigerators",
    "campsite_id": 3,
    "campsite": SLOT_COMMUNAL["campsite"],
    "distance": -0.88,
}


def _claim(campsite_id, *, is_positive, distance=-0.85, days_ago=10, text="fridge"):
    return {
        "claim": text,
        "campsite_id": campsite_id,
        "is_positive": is_positive,
        "date": "2026-08-23",
        "days_ago": days_ago,
        "distance": distance,
    }


def _constraints_state(constraints: dict) -> dict:
    return {
        "messages": [
            HumanMessage(content="מקום עם מקררים"),
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


def _plan(semantic: list) -> dict:
    return _fits_payload(
        planner_node(
            _constraints_state(
                {
                    "date": DATE,
                    "numeric_constraints": [],
                    "semantic_constraints": semantic,
                }
            )
        )
    )


def _by_site(payload: dict, field: str = "fits") -> dict[int, dict]:
    return {row["campsite_id"]: row for row in payload[field]}


@pytest.fixture
def fridge_db(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr("source.agent.search._query_vec_literal", lambda query: FAKE_VEC)
    slots = MagicMock(return_value=[dict(SLOT_COMMUNAL), dict(SLOT_IN_ROOM)])
    amenities = MagicMock(return_value=[dict(TYPE_FRIDGE_HIT)])
    site_amenities = MagicMock(return_value=[dict(SITE_FRIDGE_HIT)])
    claims = MagicMock(return_value=[])
    lookup = MagicMock(return_value=[])
    campsites = MagicMock(return_value=[])
    monkeypatch.setattr("source.agent.search.search_open_slots", slots)
    monkeypatch.setattr("source.agent.search.search_stated_amenities", amenities)
    monkeypatch.setattr("source.agent.search.search_site_amenities", site_amenities)
    monkeypatch.setattr("source.agent.search.search_review_claims", claims)
    monkeypatch.setattr("source.agent.search.lookup_campsite_by_name", lookup)
    monkeypatch.setattr("source.agent.search.search_campsites", campsites)
    return SimpleNamespace(
        slots=slots,
        amenities=amenities,
        site_amenities=site_amenities,
        claims=claims,
        lookup=lookup,
        campsites=campsites,
    )


# ---- Locus normalization (no DB, no LLM) ----


def test_normalize_locus_aliases():
    for raw in ("room", "in_room", "in the room", "unit", "private", "ROOM"):
        assert normalize_locus(raw) == "room", raw
    for raw in (None, "", "site", "campsite", "nonsense"):
        assert normalize_locus(raw) == "site", raw


def test_normalize_constraints_keeps_locus():
    out = normalize_constraints(
        {
            "date": DATE,
            "semantic_constraints": [
                {"query": "fridge", "locus": "room"},
                {"query": "fridge"},
                {"op": "or", "values": ["near the sea", "a lake"], "locus": "site"},
            ],
        }
    )
    assert out["semantic_constraints"] == [
        {"query": "fridge", "locus": "room"},
        {"query": "fridge", "locus": "site"},
        {"op": "or", "values": ["near the sea", "a lake"], "locus": "site"},
    ]


def test_same_query_at_two_loci_stays_two_and_groups():
    """"a fridge somewhere AND one in the room" must not dedupe to one item."""
    out = normalize_constraints(
        {
            "date": DATE,
            "semantic_constraints": [
                {"query": "fridge"},
                {"query": "fridge", "locus": "room"},
            ],
        }
    )
    assert len(out["semantic_constraints"]) == 2
    groups = semantic_locus_groups(out["semantic_constraints"])
    assert [g["locus"] for g in groups] == ["site", "room"]
    assert [g["label"] for g in groups] == ["fridge", "fridge"]


def test_semantic_locus_groups_or_group_carries_one_locus():
    groups = semantic_locus_groups(
        [{"op": "or", "values": ["near the sea", "a lake"], "locus": "room"}]
    )
    assert groups == [
        {
            "queries": ["near the sea", "a lake"],
            "locus": "room",
            "label": ["near the sea", "a lake"],
        }
    ]


# ---- The fridge cases ----


def test_general_fridge_matches_communal_site_fridge(fridge_db: SimpleNamespace):
    payload = _plan([{"query": "fridge", "locus": "site"}])
    fits = _by_site(payload)
    assert set(fits) == {3, 4}, payload["rejected"]
    assert fits[3]["why"] == [
        {
            "query": "fridge",
            "site_amenity": "communal_refrigerators",
            "distance": -0.88,
        }
    ]
    assert fits[4]["why"][0]["stated_amenity"] == "refrigerator"


def test_in_room_fridge_rejects_communal_site_fridge(fridge_db: SimpleNamespace):
    payload = _plan([{"query": "fridge", "locus": "room"}])
    assert list(_by_site(payload)) == [4]
    rejected = _by_site(payload, "rejected")
    assert list(rejected) == [3]
    assert rejected[3]["why"] == [
        {"reason": "missing_room_amenity", "query": "fridge", "locus": "room"}
    ]
    # The site lane must not be consulted at all for a room ask.
    fridge_db.site_amenities.assert_not_called()


def test_missing_locus_behaves_like_site(fridge_db: SimpleNamespace):
    assert _plan([{"query": "fridge"}])["fits"] == _plan(
        [{"query": "fridge", "locus": "site"}]
    )["fits"]


def test_site_amenity_search_scoped_to_still_unmatched_sites(
    fridge_db: SimpleNamespace,
):
    _plan([{"query": "fridge", "locus": "site"}])
    # Campsite 4 is already satisfied by its own type amenity, so only 3 is asked.
    fridge_db.site_amenities.assert_called_once_with(
        "fridge", limit=1, embedding=FAKE_VEC, campsite_ids=[3]
    )


def test_claims_scoped_to_candidate_campsites(fridge_db: SimpleNamespace):
    _plan([{"query": "fridge", "locus": "site"}])
    fridge_db.claims.assert_called_once_with(
        "fridge", limit=5, embedding=FAKE_VEC, campsite_ids=[3, 4]
    )


# ---- Claims as a match lane ----


def test_positive_claim_satisfies_without_any_stated_amenity(
    fridge_db: SimpleNamespace,
):
    fridge_db.amenities.return_value = []
    fridge_db.site_amenities.return_value = []
    fridge_db.claims.return_value = [_claim(3, is_positive=True, distance=-0.75)]
    payload = _plan([{"query": "fridge", "locus": "site"}])
    fits = _by_site(payload)
    assert list(fits) == [3]
    assert fits[3]["why"][0]["claim"] == "fridge"
    assert fits[3]["why"][0]["is_positive"] is True


def test_positive_claim_satisfies_a_room_locus_too(fridge_db: SimpleNamespace):
    fridge_db.amenities.return_value = []
    fridge_db.claims.return_value = [_claim(3, is_positive=True, distance=-0.75)]
    payload = _plan([{"query": "fridge", "locus": "room"}])
    fits = _by_site(payload)
    assert list(fits) == [3]
    assert fits[3]["why"][0]["locus"] == "room"


def test_negative_claim_never_vetoes_a_stated_amenity(fridge_db: SimpleNamespace):
    fridge_db.claims.return_value = [_claim(4, is_positive=False, distance=-0.95)]
    payload = _plan([{"query": "fridge", "locus": "room"}])
    fits = _by_site(payload)
    assert 4 in fits
    assert fits[4]["why"][0]["stated_amenity"] == "refrigerator"
    assert fits[4]["review_claims"][0]["is_positive"] is False


def test_negative_claim_alone_does_not_satisfy(fridge_db: SimpleNamespace):
    fridge_db.amenities.return_value = []
    fridge_db.site_amenities.return_value = []
    fridge_db.claims.return_value = [_claim(3, is_positive=False, distance=-0.95)]
    payload = _plan([{"query": "fridge", "locus": "site"}])
    assert payload["fits"] == []
    assert payload["rejected_count"] == 2


def test_neutral_claim_does_not_satisfy(fridge_db: SimpleNamespace):
    fridge_db.amenities.return_value = []
    fridge_db.site_amenities.return_value = []
    fridge_db.claims.return_value = [_claim(3, is_positive=None, distance=-0.95)]
    assert _plan([{"query": "fridge", "locus": "site"}])["fits"] == []


def test_positive_claim_outside_threshold_does_not_satisfy(
    fridge_db: SimpleNamespace,
):
    fridge_db.amenities.return_value = []
    fridge_db.site_amenities.return_value = []
    fridge_db.claims.return_value = [_claim(3, is_positive=True, distance=-0.5)]
    assert _plan([{"query": "fridge", "locus": "site"}])["fits"] == []


# ---- "AC in the room": the two ways a room ask can be satisfied ----

AC_IN_ROOM = [{"query": "AC in room", "locus": "room"}]
TYPE_AC_HIT = dict(
    TYPE_FRIDGE_HIT, amenity="air_conditioning_in_room", distance=-0.91
)


def test_stated_room_amenity_beats_contradicting_negative_claims(
    fridge_db: SimpleNamespace,
):
    """Listing says the room has AC; guests say it is absent / broken → still fits.

    Polarity cannot tell "no AC in the room" (absent) from "the AC is not
    working" (present but broken), so neither may veto the official listing.
    """
    fridge_db.amenities.return_value = [dict(TYPE_AC_HIT)]
    fridge_db.site_amenities.return_value = []
    fridge_db.claims.return_value = [
        _claim(4, is_positive=False, distance=-0.9, text="No AC in the room"),
        _claim(4, is_positive=False, distance=-0.88, text="the AC is not working"),
    ]
    payload = _plan(AC_IN_ROOM)
    fits = _by_site(payload)
    assert list(fits) == [4], payload["rejected"]
    assert fits[4]["why"][0]["stated_amenity"] == "air_conditioning_in_room"
    assert fits[4]["why"][0]["locus"] == "room"
    # Still surfaced as caveats, and they drag the ranking down.
    assert [c["claim"] for c in fits[4]["review_claims"]] == [
        "No AC in the room",
        "the AC is not working",
    ]
    assert fits[4]["score"] < 0


def test_positive_claim_alone_satisfies_a_room_amenity(fridge_db: SimpleNamespace):
    """No listing amenity anywhere; a guest says the rooms have AC → fits."""
    fridge_db.amenities.return_value = []
    fridge_db.site_amenities.return_value = []
    fridge_db.claims.return_value = [
        _claim(3, is_positive=True, distance=-0.82, text="rooms have AC")
    ]
    payload = _plan(AC_IN_ROOM)
    fits = _by_site(payload)
    assert list(fits) == [3], payload["rejected"]
    assert fits[3]["why"][0] == {
        "query": "AC in room",
        "claim": "rooms have AC",
        "date": "2026-08-23",
        "days_ago": 10,
        "distance": -0.82,
        "is_positive": True,
        "locus": "room",
    }
    assert fits[3]["score"] > 0


# ---- Ranking and evidence ----


def test_fits_ordered_by_claim_polarity(fridge_db: SimpleNamespace):
    fridge_db.amenities.return_value = [
        dict(TYPE_FRIDGE_HIT),
        dict(TYPE_FRIDGE_HIT, accommodation_type_id=11, hotel_id=3),
    ]
    fridge_db.claims.return_value = [
        _claim(3, is_positive=False, days_ago=5),
        _claim(4, is_positive=True, days_ago=5),
    ]
    payload = _plan([{"query": "fridge", "locus": "site"}])
    assert [f["campsite_id"] for f in payload["fits"]] == [4, 3]
    assert payload["fits"][0]["score"] > 0 > payload["fits"][1]["score"]


def test_recency_decays_by_half_life(fridge_db: SimpleNamespace):
    fridge_db.amenities.return_value = [
        dict(TYPE_FRIDGE_HIT),
        dict(TYPE_FRIDGE_HIT, accommodation_type_id=11, hotel_id=3),
    ]
    fridge_db.claims.return_value = [
        _claim(3, is_positive=True, days_ago=0),
        _claim(4, is_positive=True, days_ago=CLAIM_RECENCY_HALF_LIFE_DAYS),
    ]
    fits = _by_site(_plan([{"query": "fridge", "locus": "site"}]))
    assert fits[3]["score"] == pytest.approx(1.0)
    assert fits[4]["score"] == pytest.approx(0.5)


def test_fresh_positive_outweighs_stale_negative(fridge_db: SimpleNamespace):
    fridge_db.amenities.return_value = [
        dict(TYPE_FRIDGE_HIT),
        dict(TYPE_FRIDGE_HIT, accommodation_type_id=11, hotel_id=3),
    ]
    fridge_db.claims.return_value = [
        _claim(3, is_positive=True, days_ago=0, text="fridge worked"),
        _claim(3, is_positive=False, days_ago=730, text="fridge was broken"),
        _claim(4, is_positive=False, days_ago=0),
    ]
    payload = _plan([{"query": "fridge", "locus": "site"}])
    assert [f["campsite_id"] for f in payload["fits"]] == [3, 4]
    assert _by_site(payload)[3]["score"] == pytest.approx(1.0 - 0.25)


def test_no_claims_keeps_vacancy_order_and_zero_score(fridge_db: SimpleNamespace):
    payload = _plan([{"query": "fridge", "locus": "site"}])
    assert [f["campsite_id"] for f in payload["fits"]] == [3, 4]
    assert all(f["score"] == 0.0 for f in payload["fits"])
    assert all("review_claims" not in f for f in payload["fits"])


def test_evidence_includes_both_polarities_capped_at_five(
    fridge_db: SimpleNamespace,
):
    fridge_db.claims.return_value = [
        _claim(3, is_positive=True, distance=-0.99, text="a"),
        _claim(3, is_positive=True, distance=-0.98, text="b"),
        _claim(3, is_positive=True, distance=-0.97, text="c"),
        _claim(3, is_positive=True, distance=-0.96, text="d"),
        _claim(3, is_positive=True, distance=-0.95, text="e"),
        _claim(3, is_positive=True, distance=-0.94, text="f"),
        _claim(3, is_positive=False, distance=-0.93, text="but it leaked"),
    ]
    evidence = _by_site(_plan([{"query": "fridge", "locus": "site"}]))[3][
        "review_claims"
    ]
    assert len(evidence) == 5
    polarities = {c["is_positive"] for c in evidence}
    assert polarities == {True, False}
    assert evidence[-1]["claim"] == "but it leaked"
