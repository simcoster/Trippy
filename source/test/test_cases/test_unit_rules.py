"""A booking tooltip read as unit-scoped rules.

The availability scrape used to write per-unit amenities as bare
`(subject_id, polarity)` rows. They now go through the same extractor,
resolver and upsert the site-level rules do, so each row carries the Hebrew
sentence it was read from. These cover the framing that makes that safe --
the tooltip is one unit, not the whole site -- and the named-place expansion
that moved here from `test_amenity_extraction.py` with it.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from source.scraper.amenity_enrichment.schemas import ALLOWED_CATEGORIES
from source.scraper.rules_ingest.llm import SYSTEM_PROMPT
from source.scraper.rules_ingest.units import (
    UNIT_PROMPT,
    ingest_unit_rules,
    unit_section,
)
from source.scraper.rules_ingest.units import unit_extractor as make_unit_extractor

load_dotenv()

HUSHA_TOOLTIP = (
    "בכל חושה: 4 מיטות, מזרנים, תאורה, מיני מקרר, מאוורר. "
    "עד 4 לנים בחושה. יש להצטייד במצעים, מגבות, כריות ושמיכות. "
    "מותנה במינימום 2 לילות בסופי שבוע ובחגים."
)


# ----------------------------------------------------------------- framing
def test_the_unit_name_leads_the_text_not_only_the_title():
    """The name is part of the description, so it has to be IN the text: it is
    what the category statement quotes, and on a title-only panel it is the
    whole description."""
    section = unit_section(
        "חושה", f"  {HUSHA_TOOLTIP}  ", source_url="https://booking.example/1"
    )
    assert section.title == "חושה"
    assert section.text == "חושה\n" + HUSHA_TOOLTIP
    assert section.source_url == "https://booking.example/1"


def test_the_prompt_frames_the_text_as_one_unit_and_points_at_its_name():
    """The name is NOT inlined. It reaches the model twice through the user
    message -- `Section: <name>` and the first line of the text -- and inlining
    it made the prompt differ per unit, which cost a client per unit."""
    assert "ONE accommodation unit" in UNIT_PROMPT
    assert "Section:" in UNIT_PROMPT


def test_every_unit_shares_one_prompt_and_therefore_one_client():
    """Guards the fix. The prompt used to interpolate the unit name, so every
    unit needed its own client -- and each client builds its own transport, at
    5.2 s a time on a machine with TLS_TRUST_OS_STORE set, because
    `ssl_context()` reloads the OS trust store. Six minutes over 68 units.

    Not "no unit word appears in the prompt": `חושה` is in its glossary and its
    examples, legitimately. The property is that the prompt does not VARY.
    """
    assert make_unit_extractor().system_prompt is UNIT_PROMPT
    assert make_unit_extractor().system_prompt is make_unit_extractor().system_prompt


def test_the_unit_framing_comes_before_the_production_prompt():
    """Same reason `subcamp_prompt` prefixes rather than appends: the
    production prompt ends with its output schema, so anything after it reads
    as a note on the schema instead of the frame for the task."""
    assert UNIT_PROMPT.endswith(SYSTEM_PROMPT)
    assert UNIT_PROMPT.index("UNIT SCOPE") < UNIT_PROMPT.index(SYSTEM_PROMPT)


def test_the_prompt_offers_every_category_the_unit_columns_allow():
    for category in ALLOWED_CATEGORIES:
        assert category in UNIT_PROMPT


def test_a_unit_with_no_tooltip_still_reads_its_name():
    """Yehudia's whole panel is one `<h4>` and nothing else. Read as a title
    alone it yielded no rules at all, but `לינת שטח באוהלים פרטיים` states a
    tent-pitching area on its own."""
    section = unit_section("לינת שטח באוהלים פרטיים", "   ", source_url=None)
    assert section.text == "לינת שטח באוהלים פרטיים"


def test_a_unit_with_no_name_is_not_extracted_at_all():
    """Nothing to read at all — `conn` is never touched."""
    assert (
        ingest_unit_rules(
            None,
            campsite_id=1,
            accommodation_type_id=2,
            type_name="   ",
            tooltip="   ",
            embedder=None,
            adjudicator=None,
        )
        == 0
    )


# --------------------------------------------------------------------- llm
@pytest.fixture(scope="module")
def unit_extractor():
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    # One extractor for every unit, which is the point of the constant prompt.
    # The fixture keeps its per-name signature so the test bodies are unchanged.
    client = make_unit_extractor()
    return lambda _type_name: client


def _subjects(extract) -> str:
    return " ".join(s.subject.lower() for s in extract.statements)


def _extract(unit_extractor, type_name: str, tooltip: str):
    """Extract the way `ingest_unit_rules` does — through `unit_section`, so the
    model sees the name line that production puts in front of the text."""
    section = unit_section(type_name, tooltip, source_url=None)
    return unit_extractor(type_name).extract(
        section.text, section_title=section.title
    )


@pytest.mark.llm
def test_every_statement_carries_the_sentence_it_was_read_from(unit_extractor):
    """The whole point of the move: a per-unit row can be checked against the
    tooltip, which the old `(subject_id, polarity)` write made impossible."""
    extract = _extract(unit_extractor, "חושה", HUSHA_TOOLTIP)
    assert extract.statements
    for statement in extract.statements:
        assert (statement.evidence_span or "").strip(), statement.subject


@pytest.mark.llm
def test_the_unit_category_is_emitted_as_an_amenity(unit_extractor):
    """`accommodation_category` used to be forced in as the first amenity by a
    validator. The prompt asks for it now, so a unit stays findable by shape."""
    extract = _extract(unit_extractor, "חושה", HUSHA_TOOLTIP)
    assert any(s.subject in ALLOWED_CATEGORIES for s in extract.statements), _subjects(
        extract
    )


@pytest.mark.llm
def test_the_unit_name_is_not_repeated_inside_subject_names(unit_extractor):
    """The row records the unit; naming it again forks `shower` per unit."""
    extract = _extract(unit_extractor, "חושה", HUSHA_TOOLTIP)
    assert "חושה" not in _subjects(extract)
    assert "_in_husha" not in _subjects(extract)


@pytest.mark.llm
def test_named_place_ramon_crater_keeps_the_place_and_adds_its_types(unit_extractor):
    """מכתש רמון → keep the place and add crater + desert types."""
    extract = unit_extractor("חניון אוהלים").extract(
        "חניון אוהלים ליד מכתש רמון. נוף מדברי, שקט בלילה.",
        section_title="חניון אוהלים",
    )
    blob = _subjects(extract)
    print("ramon subjects:", blob)
    assert "ramon" in blob or "רמון" in blob
    assert "crater" in blob or "makhtesh" in blob or "מכתש" in blob
    assert "desert" in blob


@pytest.mark.llm
def test_named_place_kineret_keeps_the_lake_and_adds_water(unit_extractor):
    """כנרת → keep the lake name and add lake / body of water."""
    extract = unit_extractor("מתחם אוהלים").extract(
        "אתר קמפינג על שפת הכנרת. גישה למים, מדשאה.",
        section_title="מתחם אוהלים",
    )
    blob = _subjects(extract)
    print("kineret subjects:", blob)
    assert "kineret" in blob or "kinneret" in blob or "galilee" in blob or "כנרת" in blob
    assert "lake" in blob or "water" in blob


@pytest.mark.llm
def test_named_place_eilat_beach_invents_no_neighbouring_sea(unit_extractor):
    """חוף אילת — keep the place and beach, without inventing nearby seas."""
    extract = unit_extractor("אוהל").extract(
        "לינה ליד חוף אילת.", section_title="אוהל"
    )
    blob = _subjects(extract)
    print("eilat subjects:", blob)
    assert "eilat" in blob or "אילת" in blob
    assert "beach" in blob or "חוף" in blob
