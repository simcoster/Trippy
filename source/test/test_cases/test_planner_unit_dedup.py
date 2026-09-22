"""One site+type is retrieved and judged once across date windows."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ChatMessage, HumanMessage

from source.agent.booking import attach_booking_urls, booking_results_url
from source.agent.claim_judge import apply_claim_rule_judgements
from source.agent.graph import planner_node
from source.agent.planner import _semantic_why_by_slot, planner_fits_payload
from source.agent.recommender.recommend import compact_fit

SLOT = {
    "campsite_id": 8,
    "campsite": "Park A",
    "accommodation_type_id": 11,
    "accommodation_type": "tent",
    "room_count": 1,
    "max_occupancy": 4,
    "occupancy_unknown": False,
    "price_per_night": 400.0,
}


def _constraints_state(constraints: dict) -> dict:
    return {
        "messages": [
            HumanMessage(content="plan a trip"),
            AIMessage(content=json.dumps(constraints)),
        ]
    }


def _fits_payload(result: dict) -> dict:
    msg = result["messages"][0]
    assert isinstance(msg, ChatMessage)
    return json.loads(str(msg.content))


def _open_same_unit(monkeypatch) -> MagicMock:
    slots = MagicMock(
        side_effect=lambda **kwargs: [
            {
                **SLOT,
                "start": kwargs["date_range"]["start"],
                "end": kwargs["date_range"]["end"],
            }
        ]
    )
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
    return slots


def test_retrieve_runs_once_for_two_dates_of_same_unit(monkeypatch):
    claims = MagicMock(return_value=[])
    amenities = MagicMock(return_value=[])
    site_amenities = MagicMock(return_value=[])
    rules = MagicMock(return_value=[])
    monkeypatch.setattr(
        "source.agent.search._query_vec_literals", lambda qs: {q: "[0]" for q in qs}
    )
    monkeypatch.setattr("source.agent.search.search_stated_amenities", amenities)
    monkeypatch.setattr("source.agent.search.search_review_claims", claims)
    monkeypatch.setattr("source.agent.search.search_site_amenities", site_amenities)
    monkeypatch.setattr("source.agent.search.search_campsite_rules", rules)
    slots = [
        {**SLOT, "start": "2026-09-04", "end": "2026-09-06"},
        {**SLOT, "start": "2026-09-11", "end": "2026-09-13"},
    ]
    _semantic_why_by_slot(slots, [{"query": "צל", "locus": "site"}])
    assert claims.call_count == 1
    assert amenities.call_count == 1
    assert site_amenities.call_count == 1
    assert rules.call_count == 1


def test_planner_one_fit_for_same_unit_across_windows(monkeypatch):
    _open_same_unit(monkeypatch)
    w1 = {"start": "2026-09-04", "end": "2026-09-06"}
    w2 = {"start": "2026-09-11", "end": "2026-09-13"}
    payload = planner_fits_payload(
        {
            "date": w1,
            "date_windows": [w1, w2],
            "numeric_constraints": [],
            "semantic_constraints": [],
        }
    )
    assert len(payload["fits"]) == 1
    fit = payload["fits"][0]
    assert fit["accommodation_type_id"] == 11
    assert fit["start"] == "2026-09-04"
    assert [d["start"] for d in fit["dates"]] == ["2026-09-04", "2026-09-11"]
    assert [d["end"] for d in fit["dates"]] == ["2026-09-06", "2026-09-13"]


def test_planner_keeps_two_fits_for_two_types(monkeypatch):
    monkeypatch.setattr(
        "source.agent.search.search_open_slots",
        MagicMock(
            return_value=[
                {**SLOT, "start": "2026-09-04", "end": "2026-09-06"},
                {
                    **SLOT,
                    "accommodation_type_id": 22,
                    "accommodation_type": "cabin",
                    "start": "2026-09-04",
                    "end": "2026-09-06",
                },
            ]
        ),
    )
    monkeypatch.setattr(
        "source.agent.search.search_stated_amenities", MagicMock(return_value=[])
    )
    monkeypatch.setattr(
        "source.agent.search.search_review_claims", MagicMock(return_value=[])
    )
    monkeypatch.setattr(
        "source.agent.search.lookup_campsite_by_name", MagicMock(return_value=[])
    )
    payload = planner_fits_payload(
        {
            "date": {"start": "2026-09-04", "end": "2026-09-06"},
            "numeric_constraints": [],
            "semantic_constraints": [],
        }
    )
    assert [
        (f["accommodation_type_id"], f["start"]) for f in payload["fits"]
    ] == [(11, "2026-09-04"), (22, "2026-09-04")]


def test_judge_runs_once_for_two_dates_of_same_site():
    calls: list[tuple[str, str]] = []

    def _judge(*, query, campsite, claims, rules, usage=None, **_k):
        calls.append((campsite, query))
        return {
            "relevant_claims": [],
            "satisfies": True,
            "satisfy_by": "rules",
            "reason": "ok",
        }

    fit = {
        "campsite_id": 8,
        "campsite": "Park A",
        "accommodation_type": "tent",
        "start": "2026-09-04",
        "end": "2026-09-06",
        "why": [{"query": "צל", "site_amenity": "צל"}],
        "review_claims": [],
        "campsite_rules": {"צל": [{"subject": "shade", "polarity": True}]},
    }
    later = {**fit, "start": "2026-09-11", "end": "2026-09-13"}
    out = apply_claim_rule_judgements(
        {"fits": [fit, later], "rejected": [], "rejected_count": 0},
        judge=_judge,
    )
    assert len(calls) == 1
    assert len(out["fits"]) == 2
    assert all(f["claim_judge"][0]["satisfies"] is True for f in out["fits"])


def test_planner_node_emits_one_fit_for_two_windows(monkeypatch):
    slots = _open_same_unit(monkeypatch)
    w1 = {"start": "2026-09-04", "end": "2026-09-06"}
    w2 = {"start": "2026-09-11", "end": "2026-09-13"}
    result = planner_node(
        _constraints_state(
            {
                "date": w1,
                "date_windows": [w1, w2],
                "numeric_constraints": [],
                "semantic_constraints": [],
            }
        )
    )
    assert slots.call_count == 2
    payload = _fits_payload(result)
    assert len(payload["fits"]) == 1
    assert [d["start"] for d in payload["fits"][0]["dates"]] == [
        "2026-09-04",
        "2026-09-11",
    ]


def test_booking_url_set_on_each_night():
    fits = [
        {
            "campsite_id": 37,
            "start": "2026-09-04",
            "end": "2026-09-06",
            "dates": [
                {"start": "2026-09-04", "end": "2026-09-06"},
                {"start": "2026-09-11", "end": "2026-09-13"},
            ],
        }
    ]
    attach_booking_urls(fits, {}, hotel_ids={37: "8_1"})
    assert fits[0]["booking_url"] == booking_results_url(
        "8_1", "2026-09-04", "2026-09-06"
    )
    assert [d["booking_url"] for d in fits[0]["dates"]] == [
        booking_results_url("8_1", "2026-09-04", "2026-09-06"),
        booking_results_url("8_1", "2026-09-11", "2026-09-13"),
    ]


def test_compact_fit_keeps_dates():
    compact = compact_fit(
        {
            "campsite_id": 8,
            "campsite": "Park A",
            "accommodation_type": "tent",
            "start": "2026-09-04",
            "end": "2026-09-06",
            "dates": [
                {
                    "start": "2026-09-04",
                    "end": "2026-09-06",
                    "price_per_night": 400.0,
                }
            ],
            "score": 0.1,
        }
    )
    assert compact["dates"][0]["start"] == "2026-09-04"
    assert "score" not in compact
