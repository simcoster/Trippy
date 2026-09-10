"""Judge scores planner-retrieved claims/rules; it does not embed or search."""

from unittest.mock import MagicMock

from source.agent.claim_judge import apply_claim_rule_judgements
from source.agent.planner import planner_fits_payload


def _payload(*fits: dict) -> dict:
    return {"fits": list(fits), "rejected": [], "rejected_count": 0}


def test_judge_uses_rules_already_on_the_fit(monkeypatch):
    embed = MagicMock(side_effect=AssertionError("judge must not embed"))
    fetch = MagicMock(side_effect=AssertionError("judge must not search rules"))
    monkeypatch.setattr("source.agent.search._query_vec_literal", embed)
    monkeypatch.setattr("source.agent.search.search_campsite_rules", fetch)
    seen: dict = {}

    def _fn(*, query, campsite, claims, rules, usage=None):
        seen["rules"] = rules
        return {
            "relevant_claims": [],
            "satisfies": True,
            "satisfy_by": "rule",
            "reason": "tent pitch granted",
        }

    fit = {
        "campsite_id": 14,
        "campsite": "Masada",
        "why": [{"query": "camping", "stated_amenity": "tent_pitch"}],
        "campsite_rules": {
            "camping": [
                {
                    "subject": "tent_pitch",
                    "polarity": True,
                    "evidence_span": "לינת שטח",
                }
            ]
        },
    }
    out = apply_claim_rule_judgements(_payload(fit), judge=_fn)
    assert seen["rules"][0]["subject"] == "tent_pitch"
    assert out["fits"][0]["claim_judge"][0]["satisfies"] is True
    embed.assert_not_called()
    fetch.assert_not_called()


def test_planner_fetches_rules_with_the_same_query_vector(monkeypatch):
    vec = "[0.1,0.2,0.3]"
    monkeypatch.setattr("source.agent.search._query_vec_literal", lambda query: vec)
    monkeypatch.setattr(
        "source.agent.search.search_open_slots",
        MagicMock(
            return_value=[
                {
                    "campsite_id": 14,
                    "campsite": "Masada",
                    "start": "2026-09-10",
                    "end": "2026-09-11",
                    "room_count": 1,
                    "accommodation_type_id": 1,
                    "accommodation_type": "אוהל",
                    "max_occupancy": 2,
                    "occupancy_unknown": False,
                    "price_per_night": 64.0,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "source.agent.search.search_stated_amenities",
        MagicMock(
            return_value=[
                {
                    "amenity": "tent_pitch",
                    "accommodation_type_id": 1,
                    "distance": -0.9,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "source.agent.search.search_review_claims", MagicMock(return_value=[])
    )
    monkeypatch.setattr(
        "source.agent.search.search_site_amenities", MagicMock(return_value=[])
    )
    rules = MagicMock(
        return_value=[
            {
                "campsite_id": 14,
                "subject": "tent_pitch",
                "polarity": True,
                "evidence_span": "לינת שטח",
            }
        ]
    )
    monkeypatch.setattr("source.agent.search.search_campsite_rules", rules)

    payload = planner_fits_payload(
        {
            "date": {"start": "2026-09-10", "end": "2026-09-11"},
            "numeric_constraints": [],
            "semantic_constraints": [{"query": "camping"}],
        }
    )
    rules.assert_called_once()
    assert rules.call_args.kwargs["embedding"] == vec
    assert payload["fits"][0]["campsite_rules"]["camping"][0]["subject"] == (
        "tent_pitch"
    )
