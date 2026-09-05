"""The referent problem, pinned as strict xfails (PLAN 2026-09-05 "Referent field").

Each case is a statement whose sentence names WHAT the fact is about -- a rate,
a unit type, a membership -- and where the pipeline dropped that referent:
the extractor stored the fact as one about the campsite, or the judge merged
two facts with different referents because the names looked alike. Seen live
in reports/rules_ingest/2026-09-05_101304.md.

They are `xfail(strict=True)`: the suite says the problem is still there, and
flips loudly the day a referent field (or a prompt change) fixes one.
Run with `-m llm`.
"""

from __future__ import annotations

import os
import re

import pytest
from dotenv import load_dotenv

from source.scraper.rules_ingest.llm import RuleExtractorLLMClient
from source.scraper.subjects.llm import SubjectAdjudicatorLLMClient

pytestmark = pytest.mark.llm

load_dotenv()

PREDICATES = (
    "allowed", "required", "time", "fee_ils", "fee_percent", "min_age", "max_age",
    "min_nights", "max_nights", "min_occupancy", "max_occupancy", "count",
)
REFERENT_DROPPED = pytest.mark.xfail(
    strict=True, reason="the referent (rate / unit / membership) is dropped from the statement"
)


@pytest.fixture(scope="module")
def extractor() -> RuleExtractorLLMClient:
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    return RuleExtractorLLMClient()


@pytest.fixture(scope="module")
def judge() -> SubjectAdjudicatorLLMClient:
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    return SubjectAdjudicatorLLMClient()


def rules(statements):
    return [s for s in statements if s.category in (2, 3)]


# --- extractor: the referent must be in the statement --------------------------------
@REFERENT_DROPPED
def test_a_rate_age_band_is_about_the_rate_not_the_campsite(extractor):
    """`child_min_age 5` stored site-wide reads as an admission rule."""
    text = "לינת שטח באוהלים פרטיים - מבוגר: גיל 14 ומעלה\nלינת שטח באוהלים פרטיים - ילד: מגיל 5 ועד 14"
    statements = extractor.extract(text, section_title="הערות למחירון").statements
    ages = [s for s in rules(statements) if s.subject.endswith(("min_age", "max_age"))]
    assert ages, [s.subject for s in statements]
    assert all(re.search(r"tent|rate|price", s.subject) for s in ages), [s.subject for s in ages]


@REFERENT_DROPPED
def test_a_units_minimum_stay_names_the_unit(extractor):
    """Stored as `weekend_min_nights`, the bungalow's rule became everyone's."""
    text = "בונגלו עם מזגן סופי שבוע וחגים: מותנה במינימום 2 לילות"
    statements = extractor.extract(text, section_title="הערות למחירון").statements
    nights = [s for s in rules(statements) if s.subject.endswith("min_nights")]
    assert nights, [s.subject for s in statements]
    assert all("bungalow" in s.subject for s in nights), [s.subject for s in nights]


@REFERENT_DROPPED
def test_mattresses_included_in_a_tent_rental_are_not_the_site_count(extractor):
    """`mattresses_included 4` merged into `mattresses`; campsite 3 now says 4 mattresses."""
    text = "השכרת אוהל קמפינג משפחתי כולל מזרנים (עד 4 לנים): עד 5 לנים באוהל"
    statements = extractor.extract(text, section_title="הערות למחירון").statements
    mattresses = [s for s in statements if "mattress" in s.subject]
    assert mattresses, [s.subject for s in statements]
    assert all("tent" in s.subject for s in mattresses), [s.subject for s in mattresses]


@REFERENT_DROPPED
def test_a_members_only_price_is_not_the_general_fee(extractor):
    """`early_arrival_fee_percent 50` is the Matmon-subscriber price."""
    text = (
        "ביום ההגעה, הגעה מוקדמת אפשרית לאתר (ולא לחניון), על בסיס מקום פנוי "
        "ובתוספת תשלום, מנויי מטמון ב 50% מדמי כניסה יום."
    )
    statements = extractor.extract(text, section_title="שעות כניסה ויציאה").statements
    percent = [s for s in rules(statements) if s.subject.endswith("fee_percent")]
    assert percent, [s.subject for s in statements]
    assert all(re.search(r"matmon|member|subscriber", s.subject) for s in percent), [
        s.subject for s in percent
    ]


@REFERENT_DROPPED
def test_a_scope_never_follows_the_predicate(extractor):
    """`early_arrival_fee_required_for_tamun_subscribers` puts the scope last."""
    text = (
        "ביום ההגעה, הגעה מוקדמת אפשרית לאתר (ולא לחניון), על בסיס מקום פנוי "
        "ובתוספת תשלום, מנויי מטמון ללא תוספת תשלום."
    )
    statements = extractor.extract(text, section_title="שעות כניסה ויציאה").statements
    names = [s.subject for s in rules(statements)]
    assert names
    assert all(n.endswith(PREDICATES) for n in names), names


# --- judge: different referents never merge ------------------------------------------
@REFERENT_DROPPED
def test_a_huts_minimum_stay_is_not_the_bungalows(judge):
    match = judge.pick_match(
        "weekend_hut_min_nights",
        ["weekend_min_nights"],
        term_context="הערות למחירון: חושה סופי שבוע וחגים: מותנה במינימום 2 לילות",
        candidate_contexts={
            "weekend_min_nights": "הערות למחירון: בונגלו עם מזגן סופי שבוע וחגים: מותנה במינימום 2 לילות"
        },
        term_states="qualifier=2 nights",
        candidate_states={"weekend_min_nights": "qualifier=2 nights (campsite 1)"},
    )
    assert match is None


@REFERENT_DROPPED
def test_mattresses_in_a_tent_rental_do_not_merge_into_the_site_mattresses(judge):
    match = judge.pick_match(
        "mattresses_included",
        ["mattresses"],
        term_context="הערות למחירון: השכרת אוהל קמפינג משפחתי כולל מזרנים (עד 4 לנים): עד 5 לנים באוהל",
        candidate_contexts={"mattresses": "מה בחניון?: מזרנים (100)"},
        term_states="polarity=true qualifier=4 count",
        candidate_states={"mattresses": "polarity=true qualifier=100 count (campsite 1)"},
    )
    assert match is None
