"""Live adjudication on the collisions the per-site ingest report surfaced.

Each pair below was merged on a real run and lost a fact at upsert. They are
pinned here as must-NOT-merge, with the contexts the judge actually saw: the
field-kitchen pair was merged precisely because both statements quote the same
sentence, and the prompt now says identical contexts are not evidence of
sameness. What the resolver should do with these terms instead is decided
separately; this file only asserts that the merge does not happen.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from source.scraper.subjects.llm import SubjectAdjudicatorLLMClient

pytestmark = pytest.mark.llm

load_dotenv()

FIELD_KITCHEN = "מה בחניון?: מטבח שדה (1) בשלב הזה בלי גז"
LATE_CHECKOUT = (
    "שעות כניסה ויציאה: לנים המבקשים להישאר באתר לאחר השעה 12:00 ועד לסיום "
    "שעות הפעילות בשעה 17:00 נדרשים לתשלום של 50% מדמי כניסת יום לאתר"
)
LATE_CHECKOUT_UNITS = (
    "שעות כניסה ויציאה: ביחידות האירוח (חדרים, חושות, בונגלו) תתאפשר יציאה "
    "מאוחרת בתוספת תשלום ועל בסיס מקום פנוי"
)


@pytest.fixture(scope="module")
def adjudicator() -> SubjectAdjudicatorLLMClient:
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    return SubjectAdjudicatorLLMClient()


@pytest.mark.parametrize(
    ("term", "term_context", "candidates"),
    [
        # A part of an amenity is not the amenity, even when one list item
        # names both. This is the merge that dropped "no gas" on campsite 1.
        ("gas_stove_in_field_kitchen", FIELD_KITCHEN, {"field_kitchen": FIELD_KITCHEN}),
        ("gas_in_field_kitchen", FIELD_KITCHEN, {"field_kitchen": FIELD_KITCHEN}),
        # "A fee applies" (boolean) is not "what percent" (numeric). This merge
        # left site 20 with a fee_percent row whose value is True. The judge
        # still merges it under the current prompt; strict xfail so the suite
        # says so and flips loudly when it is fixed -- or when the modelling
        # decision makes these one subject (`late_check_out_fee` + a unit).
        pytest.param(
            "late_check_out_fee_applies",
            LATE_CHECKOUT,
            {"late_check_out_fee_percent": LATE_CHECKOUT},
            marks=pytest.mark.xfail(
                strict=True,
                reason="judge still merges a boolean fee into a percent fee; "
                "fee-as-subject vs fee-as-value modelling pending",
            ),
        ),
        # A deadline is not a permission, however shared the words.
        (
            "late_check_out_end_time",
            LATE_CHECKOUT,
            {"late_check_out_allowed": LATE_CHECKOUT},
        ),
        # A rule scoped to accommodation units is not the site-wide price.
        (
            "late_check_out_in_accommodation_units_fee_required",
            LATE_CHECKOUT_UNITS,
            {"late_check_out_fee_percent": LATE_CHECKOUT},
        ),
    ],
)
def test_collisions_seen_on_live_runs_do_not_merge(
    adjudicator, term, term_context, candidates
):
    match = adjudicator.pick_match(
        term,
        list(candidates),
        term_context=term_context,
        candidate_contexts=candidates,
    )
    assert match is None, f"{term!r} merged into {match!r}"
