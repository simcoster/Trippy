"""Judge drops are named in why_not, which was built before the judge."""

from source.agent.claim_judge import apply_claim_rule_judgements


def _judge(*, satisfies: bool):
    def _fn(**kwargs):
        return {
            "relevant_claims": [],
            "satisfies": satisfies,
            "satisfy_by": None,
            "reason": "no fridge",
        }

    return _fn


def test_judge_drop_is_named_in_why_not():
    kept = {
        "campsite_id": 1,
        "campsite": "Hurshat Tal",
        "why": [{"query": "fridge", "stated_amenity": "refrigerator"}],
    }
    dropped = {
        "campsite_id": 2,
        "campsite": "Dry Site",
        "why": [{"query": "fridge", "stated_amenity": "cooler"}],
    }
    out = apply_claim_rule_judgements(
        {"fits": [kept, dropped], "rejected": [], "rejected_count": 0},
        judge=_judge(satisfies=False),
        search_rules=lambda *a, **k: [],
    )
    # The fake judge rejects every fit, including the one we called kept.
    assert out["fits"] == []
    assert out["why_not"][0]["sites"] == ["Hurshat Tal", "Dry Site"]


def test_forbidding_rule_is_its_own_why_not_line():
    dropped = {
        "campsite_id": 2,
        "campsite": "No Pets",
        "why": [{"query": "pet friendly"}],
        "campsite_rules": {
            "pet friendly": [{"subject": "dogs_allowed", "polarity": False}]
        },
    }
    out = apply_claim_rule_judgements(
        {"fits": [dropped], "rejected": [], "rejected_count": 0},
        judge=_judge(satisfies=False),
        search_rules=lambda *a, **k: [],
    )
    assert out["why_not"] == [
        {
            "stage": "rule",
            "count": 1,
            "query": "pet friendly",
            "sites": ["No Pets"],
        }
    ]


def test_price_line_stays_when_the_judge_drops_someone_else():
    survivor = {
        "campsite_id": 1,
        "campsite": "Hurshat Tal",
        "why": [{"query": "fridge", "stated_amenity": "refrigerator"}],
    }
    dropped = {
        "campsite_id": 2,
        "campsite": "Dry Site",
        "why": [{"query": "fridge"}],
    }

    def _fn(**kwargs):
        satisfies = kwargs.get("campsite") == "Hurshat Tal"
        return {
            "relevant_claims": [],
            "satisfies": satisfies,
            "satisfy_by": "rule" if satisfies else None,
            "reason": "ok" if satisfies else "no fridge",
        }

    out = apply_claim_rule_judgements(
        {
            "fits": [survivor, dropped],
            "rejected": [],
            "rejected_count": 0,
            "why_not": [
                {
                    "stage": "missing",
                    "count": 1,
                    "query": "fridge",
                    "sites": ["Already Dry"],
                }
            ],
        },
        judge=_fn,
        search_rules=lambda *a, **k: [],
    )
    assert [fit["campsite"] for fit in out["fits"]] == ["Hurshat Tal"]
    assert out["why_not"] == [
        {
            "stage": "missing",
            "count": 2,
            "query": "fridge",
            "sites": ["Already Dry", "Dry Site"],
        }
    ]
