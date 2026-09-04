"""Live extraction over the real Hurshat Tal sections, against hand-read gold.

One 235B call per section, shared across the module. Run with `-m llm`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from db.models import QualifierUnit, SubjectCategory
from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.rules_ingest.llm import RuleExtractorLLMClient
from source.scraper.rules_ingest.schemas import RuleStatement
from source.scraper.rules_ingest.sections import parse_sections
from source.scraper.subjects.naming import to_positive_subject

pytestmark = pytest.mark.llm

load_dotenv()

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "info_site"
    / "hurshat_tal_camping.html"
)


@pytest.fixture(scope="module")
def statements() -> list[RuleStatement]:
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    extractor = RuleExtractorLLMClient()
    usage = LlmUsage()
    out: list[RuleStatement] = []
    for section in parse_sections(FIXTURE.read_text(encoding="utf-8")):
        extract = extractor.extract(
            section.text, section_title=section.title, usage=usage
        )
        out.extend(extract.statements)
    # summary() carries a "≈", which a cp1252 console cannot encode.
    print(usage.summary(prefix="Rules extraction: ").encode("ascii", "replace").decode())
    return out


def find(statements: list[RuleStatement], *needles: str) -> RuleStatement | None:
    """First statement whose subject contains every needle."""
    for statement in statements:
        subject = statement.subject.casefold()
        if all(n in subject for n in needles):
            return statement
    return None


def test_something_was_extracted(statements):
    assert statements


def test_no_subject_is_negatively_phrased(statements):
    """The prompt's first job. A failure here means polarity leaked into a name."""
    offenders = [
        s.subject for s in statements if to_positive_subject(s.subject)[1] is not None
    ]
    assert not offenders, f"negatively phrased subjects: {offenders}"


def test_dogs_are_recorded_as_not_allowed(statements):
    statement = find(statements, "dog")
    assert statement is not None, [s.subject for s in statements]
    assert statement.polarity is False


def test_arrival_window_lands_as_decimal_hours(statements):
    """20:30 has to survive as 20.5 — the qualifier is numeric, not a time."""
    check_in = find(statements, "check_in")
    assert check_in is not None
    assert float(check_in.qualifier) == 15.0
    assert check_in.qualifier_unit == int(QualifierUnit.HOUR_OF_DAY)

    # The canonical shape names the window's close `check_in_end_time`; older
    # extractions said `latest_arrival_time`. Either is the 20:30 fact.
    latest = (
        find(statements, "check_in_end")
        or find(statements, "arrival")
        or find(statements, "latest")
    )
    assert latest is not None
    assert float(latest.qualifier) == 20.5
    assert latest.qualifier_unit == int(QualifierUnit.HOUR_OF_DAY)


def test_checkout_time_is_extracted(statements):
    statement = find(statements, "check_out")
    assert statement is not None
    assert float(statement.qualifier) == 12.0
    assert statement.qualifier_unit == int(QualifierUnit.HOUR_OF_DAY)


def test_weekend_minimum_stay_is_extracted_in_nights(statements):
    statement = find(statements, "night")
    assert statement is not None
    assert float(statement.qualifier) == 2.0
    assert statement.qualifier_unit == int(QualifierUnit.NIGHTS)


def test_adult_age_threshold_is_extracted_in_years(statements):
    statement = find(statements, "age")
    assert statement is not None
    assert float(statement.qualifier) == 14.0
    assert statement.qualifier_unit == int(QualifierUnit.YEARS)


def test_site_amenity_counts_are_kept(statements):
    """`מקררים (11)` is worth more than a bare "has fridges"."""
    fridge = find(statements, "refrigerator") or find(statements, "fridge")
    assert fridge is not None
    assert float(fridge.qualifier) == 11.0
    assert fridge.qualifier_unit == int(QualifierUnit.COUNT)


@pytest.mark.parametrize(
    ("needle", "category"),
    [
        ("dog", SubjectCategory.BOOLEAN_RULE),
        ("check_in", SubjectCategory.NUMERIC_RULE),
        ("check_out", SubjectCategory.NUMERIC_RULE),
        ("night", SubjectCategory.NUMERIC_RULE),
        ("age", SubjectCategory.NUMERIC_RULE),
        ("refrigerator", SubjectCategory.AMENITY),
        ("shower", SubjectCategory.AMENITY),
    ],
)
def test_statements_are_categorised(statements, needle, category):
    """The category keeps a rule out of an amenity's candidate list, and a
    permission out of a deadline's."""
    statement = find(statements, needle)
    assert statement is not None, [s.subject for s in statements]
    assert statement.category == int(category)


def test_one_sentence_can_yield_a_rule_and_an_amenity(statements):
    """"ניתן להדליק מנגל בציוד עצמי" is a permission AND a missing amenity."""
    allowed = find(statements, "barbecue", "allowed")
    equipment = find(statements, "barbecue", "equipment")
    assert allowed is not None and equipment is not None
    assert allowed.category == int(SubjectCategory.BOOLEAN_RULE)
    assert allowed.polarity is True
    assert equipment.category == int(SubjectCategory.AMENITY)
    assert equipment.polarity is False


def test_every_statement_asserts_something(statements):
    """A row with neither a polarity nor a number cannot answer any question."""
    empty = [
        s.subject
        for s in statements
        if s.polarity is None and s.qualifier is None
    ]
    assert not empty, f"assert nothing: {empty}"


def test_a_qualitative_fact_becomes_a_boolean(statements):
    """"שעות פתיחה: על פי צורך" is "keeps regular hours? no"."""
    hours = [
        s
        for s in statements
        if "service_center" in s.subject and s.qualifier is None
    ]
    assert hours, [s.subject for s in statements if "service" in s.subject]
    assert all(s.polarity is not None for s in hours)


def test_no_subject_is_stated_twice(statements):
    """One row per subject per site — a repeat is silently discarded at upsert."""
    seen: dict[str, int] = {}
    for statement in statements:
        seen[statement.subject] = seen.get(statement.subject, 0) + 1
    assert not [s for s, n in seen.items() if n > 1], seen


def test_a_time_range_becomes_two_named_subjects(statements):
    """Mattress pickup is 15:00-20:00; one qualifier column cannot hold both."""
    pickup = [
        s
        for s in statements
        if "mattress" in s.subject and s.qualifier_unit == int(QualifierUnit.HOUR_OF_DAY)
    ]
    assert len(pickup) >= 2, [s.subject for s in statements if "mattress" in s.subject]
    assert len({s.subject for s in pickup}) == len(pickup)


def test_every_statement_cites_its_source_sentence(statements):
    missing = [s.subject for s in statements if not (s.evidence_span or "").strip()]
    assert not missing, f"no evidence span for: {missing}"
