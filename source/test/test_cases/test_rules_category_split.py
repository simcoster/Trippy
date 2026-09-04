"""Three subject categories: amenity, boolean rule, numeric rule.

The no-token tests pin the plumbing: the extractor's labels reach the resolver
as the right integers, the classifier's payload accepts the third value, and
the prompt carries the split and the numeric-range example. The `llm` tests
put real sentences through the 235B and check that a boolean and a numeric
statement about one topic come back on different shelves, and that a range
like 20-50 becomes a min AND a max.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from db.models import QualifierUnit, SubjectCategory
from source.scraper.rules_ingest.llm import SYSTEM_PROMPT, RuleExtractorLLMClient
from source.scraper.rules_ingest.schemas import RuleStatement
from source.scraper.subjects.resolve import category_label
from source.scraper.subjects.schemas import ClassificationPayload

load_dotenv()


# --- the enum ---------------------------------------------------------------
def test_the_three_categories_are_distinct_integers():
    assert (
        int(SubjectCategory.AMENITY),
        int(SubjectCategory.BOOLEAN_RULE),
        int(SubjectCategory.NUMERIC_RULE),
    ) == (1, 2, 3)


def test_rule_is_an_alias_of_boolean_rule():
    """Old callers still import RULE; the canonical name is the new one."""
    assert SubjectCategory.RULE is SubjectCategory.BOOLEAN_RULE
    assert SubjectCategory(2).name == "BOOLEAN_RULE"
    assert category_label(3) == "numeric_rule"


# --- the extractor payload ----------------------------------------------------
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("amenity", 1),
        ("boolean_rule", 2),
        ("Boolean_Rule", 2),
        ("numeric_rule", 3),
        ("numeric", 3),
        (3, 3),
        ("3", 3),
        # A bare "rule" no longer says which kind: search everything.
        ("rule", None),
        (7, None),
        ("", None),
        (None, None),
    ],
)
def test_extractor_category_labels_coerce(label, expected):
    statement = RuleStatement(subject="x", category=label, polarity=True)
    assert statement.category == expected


# --- the classifier payload ----------------------------------------------------
@pytest.mark.parametrize(
    ("label", "expected"),
    [("numeric_rule", 3), (3, 3), ("boolean_rule", 2), ("rule", 2), ("amenity", 1), (9, 1)],
)
def test_classifier_category_coerces(label, expected):
    payload = ClassificationPayload(category=label, canonical_name="x")
    assert payload.category == expected


# --- the prompt ---------------------------------------------------------------
def test_prompt_names_all_three_categories_and_no_bare_rule_example():
    assert '"boolean_rule"' in SYSTEM_PROMPT and '"numeric_rule"' in SYSTEM_PROMPT
    bare = [line for line in SYSTEM_PROMPT.splitlines() if "/ rule /" in line]
    assert not bare, bare


def test_prompt_shows_a_numeric_range_as_min_and_max():
    assert "30-80" in SYSTEM_PROMPT
    assert "_min_occupancy / numeric_rule / null / 30 / count" in SYSTEM_PROMPT
    assert "_max_occupancy / numeric_rule / null / 80 / count" in SYSTEM_PROMPT
    assert "min_occupancy  max_occupancy" in SYSTEM_PROMPT  # in the predicate list


# --- live extraction ------------------------------------------------------------
@pytest.fixture(scope="module")
def extractor() -> RuleExtractorLLMClient:
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    return RuleExtractorLLMClient()


def _find(statements, *needles):
    for s in statements:
        if all(n in s.subject for n in needles):
            return s
    return None


@pytest.mark.llm
def test_a_numeric_range_becomes_a_min_and_a_max(extractor):
    """Not the prompt's own 30-80 example: a different range, same shape."""
    text = "לינה לקבוצות מאורגנות (20-50 משתתפים המשלמים יחד): בתיאום מראש"
    statements = extractor.extract(text, section_title="מערכת הזמנות").statements
    numeric = [s for s in statements if s.qualifier_unit == int(QualifierUnit.COUNT)]
    values = sorted(float(s.qualifier) for s in numeric if s.qualifier is not None)
    assert values == [20.0, 50.0], [(s.subject, s.qualifier) for s in statements]
    low = _find(numeric, "min")
    high = _find(numeric, "max")
    assert low is not None and high is not None, [s.subject for s in numeric]
    assert float(low.qualifier) == 20.0 and float(high.qualifier) == 50.0
    assert {low.category, high.category} == {int(SubjectCategory.NUMERIC_RULE)}
    # One topic, two ends: the names differ only by the direction word.
    assert low.subject.replace("min", "") == high.subject.replace("max", "")


@pytest.mark.llm
def test_a_permission_and_a_deadline_land_on_different_shelves(extractor):
    """The pair the judge merged in production; the categories keep them apart now."""
    text = (
        "ניתן להשאיר את הרכב בחניון עד השעה 16:00 ביום העזיבה, "
        "בתוספת תשלום של 20 ש\"ח."
    )
    statements = extractor.extract(text, section_title="שעות כניסה ויציאה").statements
    booleans = [s for s in statements if s.category == int(SubjectCategory.BOOLEAN_RULE)]
    numerics = [s for s in statements if s.category == int(SubjectCategory.NUMERIC_RULE)]
    assert booleans, [(s.subject, s.category) for s in statements]
    assert numerics, [(s.subject, s.category) for s in statements]
    assert all(s.polarity is not None for s in booleans)
    assert all(s.qualifier is not None for s in numerics)
    hour = _find(numerics, "time")
    assert hour is not None and float(hour.qualifier) == 16.0
