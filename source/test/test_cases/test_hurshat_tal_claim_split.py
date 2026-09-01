"""Hurshat Tal claim split vs locked gold. Live 235B split + 30B judge."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_30B_MODEL,
    _parse_json_payload,
    make_nebius_openai_client,
)
from source.scraper.populate_reviews_and_claims import (
    normalize_review_dict,
    split_one_review,
)

load_dotenv()

pytestmark = pytest.mark.llm

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "reviews"
    / "hurshat_tal_most_relevant.json"
)

PLACE = "חורשת טל"

JUDGE_SYSTEM = """You judge whether a new claim split still covers a locked gold split.

Gold and candidate are lists of {claim, polarity} for THE SAME review.
Wording may differ. Park / site names may be present or absent.

gold_recall: fraction of gold items that have a candidate partner
(same polarity, same site fact). 1.0 means every gold fact is still there.

match is true iff gold_recall is 1.0 AND none of the fail conditions below.

Extra candidate claims do NOT fail the match when they are:
- supporting beats of a gold incident (asked again, cars turned around,
  cashier repeated the rule), or
- other facts clearly in THIS review that gold did not list as separate rows

match is false if:
- a gold fact is missing (gold_recall < 1)
- a paired claim has the opposite polarity
- candidate invents something not in the review
- candidate emits a generic overall judgment gold omitted
  ("disappointment", "great place", "shameful")

Output valid JSON only:
{"match": bool, "gold_recall": number, "reason": str}
"""


def _gold_rows(raw: dict) -> list[dict]:
    return [
        {"claim": item["claim"], "polarity": item["polarity"]}
        for item in raw.get("gold_claims") or []
    ]


def _candidate_rows(claims: list[dict]) -> list[dict]:
    return [
        {"claim": item["text_en"], "polarity": item.get("polarity")}
        for item in claims
    ]


def _recall(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def judge_split(
    client: object,
    *,
    review_index: int,
    gold: list[dict],
    candidate: list[dict],
) -> dict:
    user = json.dumps(
        {
            "review_index": review_index,
            "n_gold": len(gold),
            "n_candidate": len(candidate),
            "gold": gold,
            "candidate": candidate,
        },
        ensure_ascii=False,
    )
    response = client.chat.completions.create(
        model=QWEN_INSTRUCT_30B_MODEL,
        temperature=0,
        max_tokens=500,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    parsed = _parse_json_payload((response.choices[0].message.content or "").strip())
    return {
        "match": bool(parsed.get("match")),
        "gold_recall": _recall(parsed.get("gold_recall")),
        "reason": str(parsed.get("reason") or ""),
    }


@pytest.fixture(scope="module")
def nebius_client():
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    return make_nebius_openai_client()


def test_hurshat_tal_split_matches_gold(nebius_client, record_property):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    reviews = payload["reviews"]
    assert len(reviews) == 5
    failures: list[str] = []
    recalls: list[float] = []
    for i, raw in enumerate(reviews, 1):
        gold = _gold_rows(raw)
        review = normalize_review_dict(raw)
        print(
            f"splitting review {i}/5 "
            f"(gold n={len(gold)} polarities={[g['polarity'] for g in gold]})"
        )
        candidate_claims = split_one_review(
            nebius_client, review, place=PLACE
        )
        candidate = _candidate_rows(candidate_claims)
        print(
            f"  candidate n={len(candidate)} "
            f"polarities={[c['polarity'] for c in candidate]}"
        )
        verdict = judge_split(
            nebius_client, review_index=i, gold=gold, candidate=candidate
        )
        recalls.append(verdict["gold_recall"])
        record_property(f"review_{i}_gold_recall", f"{verdict['gold_recall']:.3f}")
        print(
            f"  judge match={verdict['match']} "
            f"gold_recall={verdict['gold_recall']:.3f}: {verdict['reason']}"
        )
        if not verdict["match"]:
            failures.append(
                f"review {i}: gold n={len(gold)} candidate n={len(candidate)} "
                f"gold_recall={verdict['gold_recall']:.3f} — {verdict['reason']}"
            )
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    record_property("mean_gold_recall", f"{mean_recall:.3f}")
    print(f"mean gold_recall={mean_recall:.3f}")
    assert not failures, "\n".join(failures)
