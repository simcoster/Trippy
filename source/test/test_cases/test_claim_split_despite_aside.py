"""Claim splitter: concessive aside (despite X, Y) becomes two claims."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from source.scraper.amenity_enrichment.llm import make_nebius_openai_client
from source.scraper.populate_reviews_and_claims import (
    SPLIT_SYSTEM,
    normalize_review_dict,
    split_one_review,
)

load_dotenv()

TRIALS = range(5)

PROMPT_EN = "The tent is clean despite being in the desert with winds."
PROMPT_HE = "האוהל עצמו נקי ולא מאד מאובק (בכל זאת מדבר ורוחות)."
REVIEW_MAMSHIT = """היינו במטמון שטח בסופש 13.2 ,
הייתה חוויה ממש נעימה  תודות למנהל האתר אבו ג'ודה .
האתר עצמו בנוי מ4 חלקים :אוהל לינה משותפת  גדול,טוקטלים -אוהלים משפחתים עם חימום, אזור קמפינג חופשי והאתר ממשית.
אנחנו ישנו באוהל הגדול, שמחולק בחוצצים ככה שיש קצת פרטיות , היו כאלה שהגדילו והקימו שם אוהל פרטי קטן, האוהל עצמו נקי ולא מאד מאובק(בכל זאת מדבר ורוחות) .
המלתחות: מרחק הליכה קצר היו נקיים , מים  חמים ולחץ חזק היה ממש תענוג.
שירותים:ליד מתחם הקמפינג וליד הלתחות גם נקיים ונוקו תוך כדי הלילה.
"""


def test_split_prompt_teaches_despite_aside():
    assert "despite being in the desert with winds" in SPLIT_SYSTEM
    assert "בכל זאת מדבר ורוחות" in SPLIT_SYSTEM
    assert "emit X and Y as two claims" in SPLIT_SYSTEM


def _texts(claims: list[dict]) -> list[str]:
    return [str(c.get("text_en") or "").lower() for c in claims]


def _assert_desert_split(claims: list[dict], *, label: str) -> None:
    texts = _texts(claims)
    desert = [t for t in texts if "desert" in t]
    clean = [t for t in texts if "clean" in t or "dust" in t]
    standalone_desert = [
        t for t in desert if "clean" not in t and "dust" not in t
    ]
    assert desert, f"{label}: no desert claim in {texts!r}"
    assert clean, f"{label}: no clean/dust claim in {texts!r}"
    assert standalone_desert, (
        f"{label}: desert stayed glued to cleanliness: {texts!r}"
    )


@pytest.fixture(scope="module")
def nebius_client():
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    return make_nebius_openai_client()


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_split_english_despite_desert_and_clean_tent(trial, nebius_client):
    claims = split_one_review(
        nebius_client,
        normalize_review_dict({"text": PROMPT_EN, "rating": 5}),
        place="ממשית",
    )
    _assert_desert_split(claims, label=f"en t{trial + 1}")


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_split_hebrew_parenthetical_desert_and_clean_tent(trial, nebius_client):
    claims = split_one_review(
        nebius_client,
        normalize_review_dict({"text": PROMPT_HE, "rating": 5}),
        place="ממשית",
    )
    _assert_desert_split(claims, label=f"he t{trial + 1}")


@pytest.mark.llm
@pytest.mark.parametrize("trial", TRIALS, ids=lambda i: f"t{i + 1}")
def test_split_mamshit_review_desert_aside(trial, nebius_client):
    claims = split_one_review(
        nebius_client,
        normalize_review_dict({"text": REVIEW_MAMSHIT, "rating": 5}),
        place="חניון לילה גן לאומי ממשית – החאן הנבטי",
    )
    _assert_desert_split(claims, label=f"mamshit t{trial + 1}")
