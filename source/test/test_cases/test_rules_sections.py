"""parse_sections over a trimmed copy of the real Hurshat Tal camping page."""

from pathlib import Path

import pytest

from source.scraper.rules_ingest.sections import Section, parse_sections

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "info_site"
    / "hurshat_tal_camping.html"
)


@pytest.fixture(scope="module")
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sections(html) -> list[Section]:
    return parse_sections(html, source_url="https://www.parks.org.il/camping/x/")


def by_title(sections: list[Section], needle: str) -> Section:
    matches = [s for s in sections if needle in s.title]
    assert len(matches) == 1, f"expected exactly one {needle!r}, got {len(matches)}"
    return matches[0]


def test_the_four_rule_bearing_shapes_are_found(sections):
    titles = [s.title for s in sections]
    assert "מה בחניון?" in titles  # infoArea amenity list
    assert "כניסת כלבים" in titles  # icon strip
    assert "שעות כניסה ויציאה" in titles  # visitor-info column
    assert "הערות למחירון" in titles  # rate-card tooltips


def test_unit_level_sections_are_excluded(sections):
    """Per-accommodation-type data comes from the availability scrape."""
    assert not [s for s in sections if "אפשרויות לינה" in s.title]
    joined = "\n".join(s.text for s in sections)
    assert "יש להצטייד במצעים ומגבות" not in joined
    assert "עד 5 לנים בכל חדר" not in joined


def test_navigation_columns_are_skipped(sections):
    assert not [s for s in sections if "איך להגיע" in s.title]
    assert "04-6942440" not in "\n".join(s.text for s in sections)


def test_site_amenity_list_keeps_its_counts(sections):
    text = by_title(sections, "מה בחניון").text
    assert "ברזיות מים לשתייה (6)" in text
    assert "מקררים (11)" in text
    assert "ניתן להדליק מצלה (מנגל) בציוד עצמי" in text
    # One line per list item, so the extractor sees them as separate facts.
    assert "תאורת שטח\n" in text


def test_the_dog_policy_pairs_its_label_with_its_verdict(sections):
    assert by_title(sections, "כניסת כלבים").text == "כניסת כלבים: הכניסה לכלבים אסורה"


def test_the_dog_policy_appears_exactly_once(sections):
    """The icon strip is nested in a visitor-info column; it must not double up."""
    hits = [s for s in sections if "הכניסה לכלבים אסורה" in s.text]
    assert len(hits) == 1


def test_visitor_info_carries_the_time_rules(sections):
    text = by_title(sections, "שעות כניסה ויציאה").text
    assert "החל מהשעה 15:00 עד השעה 20:30" in text
    assert "עד השעה 12:00 ביום העזיבה" in text
    assert not text.startswith("שעות כניסה ויציאה")  # heading stripped from the body


def test_rate_notes_are_labelled_and_deduped(sections):
    text = by_title(sections, "הערות למחירון").text
    assert "לינת שטח באוהלים פרטיים - מבוגר: גיל 14 ומעלה" in text
    assert "מותנה במינימום 2 לילות" in text
    assert text.count("מותנה במינימום 2 לילות") == 1


def test_every_section_carries_the_source_url(sections):
    assert {s.source_url for s in sections} == {"https://www.parks.org.il/camping/x/"}


def test_empty_sections_are_dropped(sections):
    assert all(s.text.strip() for s in sections)


def test_a_page_with_none_of_the_structures_yields_nothing():
    assert parse_sections("<html><body><p>hello</p></body></html>") == []
