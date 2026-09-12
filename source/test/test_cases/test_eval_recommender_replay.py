"""Eval dumps a recommender pack and can replay without the planner."""

import pytest

from source.agent.recommender import Recommendation, RecommendResult
from source.eval.run import (
    payload_from_pack,
    planner_pack,
    replay_recommend_rows,
)


def _fit():
    return {
        "campsite_id": 37,
        "campsite": "אכזיב צפון",
        "accommodation_type": "אוהל",
        "start": "2026-09-10",
        "end": "2026-09-11",
        "price_per_night": 150,
        "score": 0.42,
        "why": [{"query": "near the sea", "site_amenity": "beach_access"}],
        "review_claims": [
            {
                "query": "near the sea",
                "claim": "Access to the beach is easy.",
                "is_positive": True,
                "days_ago": 40,
            }
        ],
        "claim_judge": [
            {
                "query": "near the sea",
                "satisfies": True,
                "satisfy_by": "claim",
                "reason": "beach",
            }
        ],
    }


def test_planner_pack_keeps_claims_and_drops_score():
    pack = planner_pack(
        "ליד הים",
        {"date": {"start": "2026-09-10"}},
        {"fits": [_fit()], "rejected": [{"campsite_id": 14}]},
    )
    assert pack["query"] == "ליד הים"
    assert pack["extract"]["date"]["start"] == "2026-09-10"
    assert "score" not in pack["fits"][0]
    assert pack["fits"][0]["review_claims"][0]["claim"] == (
        "Access to the beach is easy."
    )
    assert "rejected" not in pack


def test_payload_from_pack_feeds_recommend_from_payload():
    pack = planner_pack("ליד הים", {"party_size": 2}, {"fits": [_fit()]})
    payload = payload_from_pack(pack)
    assert payload["constraints"]["party_size"] == 2
    assert payload["fits"][0]["campsite_id"] == 37


def test_replay_recommend_rows_uses_pack_not_planner():
    pack = planner_pack("ליד הים", {}, {"fits": [_fit()]})
    seen: list[tuple[str, dict]] = []

    def _fake(query: str, payload: dict) -> RecommendResult:
        seen.append((query, payload))
        return RecommendResult(
            recommendations=(
                Recommendation(
                    campsite_id=37,
                    campsite="אכזיב צפון",
                    accommodation_type="אוהל",
                    start="2026-09-10",
                    end="2026-09-11",
                    price_per_night=150,
                    why="ים",
                ),
            ),
            empty=None,
            text="ים",
        )

    src = {
        "id": "E04",
        "difficulty": "easy",
        "query": "ליד הים",
        "score": {"ok": True, "failures": [], "fit_sites": [37]},
        "planner": {"fits": [{"campsite_id": 99}]},
        "pack": pack,
    }
    out = replay_recommend_rows([src], recommend_fn=_fake)
    assert seen[0][0] == "ליד הים"
    assert seen[0][1]["fits"][0]["campsite_id"] == 37
    assert out[0]["recommend"]["text"] == "ים"
    assert out[0]["score"]["fit_sites"] == [37]
    assert out[0]["planner"]["fits"][0]["campsite_id"] == 99


def test_replay_without_pack_exits():
    with pytest.raises(SystemExit, match="no recommender pack"):
        replay_recommend_rows([{"id": "E01", "query": "x"}])
