"""Planner claim/rule judge: relevant evidence vs satisfies."""

from __future__ import annotations

from source.agent.claim_judge import CLAIM_JUDGE_SYSTEM, apply_claim_rule_judgements


def test_judge_prompt_asks_for_both_decisions():
    assert "relevant_claims" in CLAIM_JUDGE_SYSTEM
    assert "satisfies" in CLAIM_JUDGE_SYSTEM
    assert "dogs_allowed with polarity false" in CLAIM_JUDGE_SYSTEM
    assert "despite being in the desert" in CLAIM_JUDGE_SYSTEM
    assert "Campfires are allowed" in CLAIM_JUDGE_SYSTEM


def _payload(*fits: dict) -> dict:
    return {"fits": list(fits), "rejected": [], "rejected_count": 0}


def _judge(*, satisfies: bool, relevant: list[str], reason: str = "x"):
    def _fn(*, query, campsite, claims, rules, usage=None):
        return {
            "relevant_claims": list(relevant),
            "satisfies": satisfies,
            "satisfy_by": "claim" if satisfies else None,
            "reason": reason,
        }

    return _fn


def test_claim_only_fit_dropped_when_judge_says_not_satisfies():
    fit = {
        "campsite_id": 4,
        "campsite": "Nahal Amud",
        "why": [{"query": "desert", "claim": "Campfires are allowed."}],
        "review_claims": [
            {"query": "desert", "claim": "Campfires are allowed.", "is_positive": True}
        ],
    }
    out = apply_claim_rule_judgements(
        _payload(fit),
        judge=_judge(satisfies=False, relevant=[], reason="unrelated"),
        search_rules=lambda *a, **k: [],
    )
    assert out["fits"] == []
    assert out["rejected_count"] == 1
    assert out["rejected"][0]["why"][-1]["reason"] == "claim_not_verified"


def test_claim_only_fit_kept_when_judge_satisfies_and_evidence_filtered():
    fit = {
        "campsite_id": 14,
        "campsite": "Masada",
        "why": [{"query": "desert", "claim": "The desert atmosphere is perfect."}],
        "review_claims": [
            {
                "query": "desert",
                "claim": "The desert atmosphere is perfect.",
                "is_positive": True,
            },
            {"query": "desert", "claim": "There is no shade at all at the site."},
        ],
    }
    out = apply_claim_rule_judgements(
        _payload(fit),
        judge=_judge(
            satisfies=True,
            relevant=["The desert atmosphere is perfect."],
        ),
        search_rules=lambda *a, **k: [],
    )
    assert len(out["fits"]) == 1
    assert [c["claim"] for c in out["fits"][0]["review_claims"]] == [
        "The desert atmosphere is perfect."
    ]
    assert out["fits"][0]["claim_judge"][0]["satisfies"] is True


def test_amenity_fit_without_claims_skips_the_judge():
    calls = {"n": 0}

    def _fn(**kwargs):
        calls["n"] += 1
        return {
            "relevant_claims": [],
            "satisfies": False,
            "satisfy_by": None,
            "reason": "should not run",
        }

    fit = {
        "campsite_id": 3,
        "campsite": "Park",
        "why": [{"query": "fridge", "stated_amenity": "refrigerator"}],
    }
    out = apply_claim_rule_judgements(
        _payload(fit),
        judge=_fn,
        search_rules=lambda *a, **k: [],
    )
    assert calls["n"] == 0
    assert out["fits"][0]["why"][0]["stated_amenity"] == "refrigerator"


def test_stated_amenity_is_not_vetoed_when_judge_says_no():
    fit = {
        "campsite_id": 4,
        "campsite": "Park",
        "why": [{"query": "fridge", "stated_amenity": "refrigerator"}],
        "review_claims": [
            {"query": "fridge", "claim": "the AC was loud", "is_positive": None}
        ],
    }
    out = apply_claim_rule_judgements(
        _payload(fit),
        judge=_judge(satisfies=False, relevant=[]),
        search_rules=lambda *a, **k: [],
    )
    assert len(out["fits"]) == 1
    assert "review_claims" not in out["fits"][0]


def test_relevant_negative_stays_on_amenity_fit():
    fit = {
        "campsite_id": 1,
        "campsite": "Horshat Tal",
        "why": [{"query": "pet friendly", "stated_amenity": "dogs_allowed"}],
        "review_claims": [
            {
                "query": "pet friendly",
                "claim": "Pets are not allowed at the site.",
                "is_positive": False,
            },
            {"query": "pet friendly", "claim": "Open fires are permitted."},
        ],
    }
    out = apply_claim_rule_judgements(
        _payload(fit),
        judge=_judge(
            satisfies=False,
            relevant=["Pets are not allowed at the site."],
        ),
        search_rules=lambda *a, **k: [],
    )
    assert len(out["fits"]) == 1
    assert [c["claim"] for c in out["fits"][0]["review_claims"]] == [
        "Pets are not allowed at the site."
    ]
