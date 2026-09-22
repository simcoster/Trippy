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


def test_why_keeps_only_claims_the_judge_named():
    fit = {
        "campsite_id": 16,
        "campsite": "Mamshit",
        "why": [
            {
                "query": "pools for children",
                "claim": "The tents are spacious.",
                "is_positive": True,
            },
            {"query": "fridge", "site_amenity": "refrigerator"},
        ],
        "review_claims": [
            {
                "query": "pools for children",
                "claim": "The tents are spacious.",
                "is_positive": True,
            }
        ],
    }

    def _fn(**kwargs):
        query = kwargs.get("query")
        if query == "fridge":
            return {
                "relevant_claims": [],
                "satisfies": True,
                "satisfy_by": "rule",
                "reason": "fridge listed",
            }
        return {
            "relevant_claims": [],
            "satisfies": False,
            "satisfy_by": None,
            "reason": "no pool",
        }

    out = apply_claim_rule_judgements(
        {"fits": [fit], "rejected": [], "rejected_count": 0},
        judge=_fn,
        search_rules=lambda *a, **k: [],
    )
    assert out["fits"] == []
    why = out["rejected"][0]["why"]
    assert why == []
    assert out["rejected"][0]["claim_judge"][0]["satisfies"] is False


def test_rules_stay_only_when_the_judge_names_them():
    fit = {
        "campsite_id": 5,
        "campsite": "Yehudiya",
        "why": [{"query": "pools for children", "stated_amenity": "pool"}],
        "campsite_rules": {
            "pools for children": [
                {"subject": "drinking_water_fountain", "polarity": True},
                {"subject": "pool", "polarity": True},
            ]
        },
    }

    def _fn(**kwargs):
        return {
            "relevant_claims": [],
            "relevant_rules": ["pool"],
            "satisfies": True,
            "satisfy_by": "rule",
            "reason": "pool listed",
        }

    out = apply_claim_rule_judgements(
        {"fits": [fit], "rejected": [], "rejected_count": 0},
        judge=_fn,
        search_rules=lambda *a, **k: [],
    )
    rules = out["fits"][0]["campsite_rules"]["pools for children"]
    assert [rule["subject"] for rule in rules] == ["pool"]
    assert out["fits"][0]["why"] == [
        {"query": "pools for children", "stated_amenity": "pool"}
    ]


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


def test_judge_drop_merges_into_an_existing_missing_line():
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


def test_price_line_stays_beside_a_judge_drop():
    survivor = {
        "campsite_id": 1,
        "campsite": "Hurshat Tal",
        "why": [{"query": "fridge"}],
    }
    dropped = {
        "campsite_id": 2,
        "campsite": "Dry Site",
        "why": [{"query": "fridge"}],
    }

    def _fn(**kwargs):
        return {
            "relevant_claims": [],
            "satisfies": kwargs.get("campsite") == "Hurshat Tal",
            "satisfy_by": None,
            "reason": "ok",
        }

    out = apply_claim_rule_judgements(
        {
            "fits": [survivor, dropped],
            "rejected": [],
            "rejected_count": 0,
            "why_not": [
                {
                    "stage": "price",
                    "count": 1,
                    "sites": ["Pricey"],
                }
            ],
        },
        judge=_fn,
        search_rules=lambda *a, **k: [],
    )
    assert out["why_not"] == [
        {"stage": "price", "count": 1, "sites": ["Pricey"]},
        {
            "stage": "missing",
            "count": 1,
            "query": "fridge",
            "sites": ["Dry Site"],
        },
    ]
