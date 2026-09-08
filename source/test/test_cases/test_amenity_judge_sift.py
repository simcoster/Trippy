"""Amenity −0.7 retrieve: the judge sifts listing hits too."""

from __future__ import annotations

from source.agent.claim_judge import apply_claim_rule_judgements
from source.agent.planner import AMENITY_MATCH_MAX_DISTANCE


def _payload(*fits: dict) -> dict:
    return {"fits": list(fits), "rejected": [], "rejected_count": 0}


def _judge(*, satisfies: bool, relevant: list[str] | None = None, reason: str = "x"):
    def _fn(*, query, campsite, claims, rules, usage=None):
        return {
            "relevant_claims": list(relevant or []),
            "satisfies": satisfies,
            "satisfy_by": "rule" if satisfies else None,
            "reason": reason,
        }

    return _fn


def test_amenity_gate_is_minus_07():
    assert AMENITY_MATCH_MAX_DISTANCE == -0.7


def test_tent_listing_dropped_when_judge_says_not_desert():
    fit = {
        "campsite_id": 3,
        "campsite": "Yehiam",
        "why": [{"query": "desert", "stated_amenity": "tent"}],
    }
    out = apply_claim_rule_judgements(
        _payload(fit),
        judge=_judge(satisfies=False, reason="tent is lodging"),
        search_rules=lambda *a, **k: [],
    )
    assert out["fits"] == []
    assert out["rejected_count"] == 1
    assert out["rejected"][0]["why"][-1]["reason"] == "claim_not_verified"


def test_outlet_listing_kept_when_judge_says_electricity():
    fit = {
        "campsite_id": 1,
        "campsite": "Horshat Tal",
        "why": [{"query": "electricity", "stated_amenity": "electric_outlet"}],
    }
    out = apply_claim_rule_judgements(
        _payload(fit),
        judge=_judge(satisfies=True, reason="bungalow outlet"),
        search_rules=lambda *a, **k: [],
    )
    assert len(out["fits"]) == 1
    assert out["fits"][0]["why"][0]["stated_amenity"] == "electric_outlet"
    assert out["fits"][0]["claim_judge"][0]["satisfies"] is True
