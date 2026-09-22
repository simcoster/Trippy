"""Turn phase line: searching, then the availability count, then ranking."""

from source.agent.planner import planner_fits_payload
from source.agent.recommender.recommend import RecommendResult, recommend_from_payload
from source.agent.turn_status import (
    found_candidates_line,
    report_turn_status,
    set_turn_status,
)

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


def test_found_line_counts_one_candidate():
    assert found_candidates_line(1) == "Found 1 candidate, filtering"
    assert found_candidates_line(3) == "Found 3 candidates, filtering"


def test_report_is_a_no_op_without_a_listener():
    report_turn_status("Searching")


def test_planner_reports_search_then_the_availability_count(monkeypatch):
    import source.agent.search.availability as availability

    seen: list[str] = []
    set_turn_status(seen.append)
    night = dict(SLOT)
    other_night = dict(
        SLOT, start="2026-05-25", end="2026-05-26"
    )
    other_site = dict(
        SLOT,
        campsite_id=2,
        campsite="Park B",
        accommodation_type_id=22,
        accommodation_type="cabin",
    )
    availability._LAST_OPEN_SLOTS_QUERY = {}
    monkeypatch.setattr(
        availability,
        "search_open_slots",
        lambda **kwargs: [night, other_night, other_site],
    )
    monkeypatch.setattr(
        availability, "quote_open_slots", lambda slots, **kwargs: slots
    )
    try:
        planner_fits_payload(
            {
                "date": {"start": "2026-05-24", "end": "2026-05-26"},
                "numeric_constraints": [],
                "semantic_constraints": [],
            }
        )
    finally:
        set_turn_status(None)

    assert seen == [
        "Searching",
        "Found 2 candidates, filtering",
    ]


def test_recommend_reports_ranking(monkeypatch):
    seen: list[str] = []
    set_turn_status(seen.append)
    monkeypatch.setattr(
        "source.agent.recommender.recommend.recommender_model",
        lambda: "test-model",
    )
    monkeypatch.setattr(
        "source.agent.recommender.recommend.primary_recommend_call",
        lambda model, chat: object(),
    )
    monkeypatch.setattr(
        "source.agent.recommender.recommend.kimi_super_fallback",
        lambda primary: None,
    )
    monkeypatch.setattr(
        "source.agent.recommender.recommend.recommend_with_fallback",
        lambda *args, **kwargs: RecommendResult(
            recommendations=(), empty=None, text=""
        ),
    )
    try:
        recommend_from_payload("a pool", {"fits": [], "constraints": {}})
    finally:
        set_turn_status(None)
    assert seen == ["Ranking"]
