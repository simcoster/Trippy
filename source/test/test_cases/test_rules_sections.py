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
    """The section is split by topic, so check across both halves."""
    halves = [s for s in sections if "שעות כניסה ויציאה" in s.title]
    text = "\n".join(s.text for s in halves)
    assert "החל מהשעה 15:00 עד השעה 20:30" in text
    assert "עד השעה 12:00 ביום העזיבה" in text
    # heading stripped from the body
    assert not halves[0].text.startswith("שעות כניסה ויציאה")


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


# --- topic split --------------------------------------------------------------


def test_arrival_and_departure_become_separate_sections(sections):
    """Whole, the model summarised: 7 facts and no early arrival in 4/5 runs.
    Split, it returned 11 facts every run. See docs/design.md."""
    halves = [s for s in sections if "שעות כניסה ויציאה" in s.title]
    assert len(halves) == 2
    arrival, departure = halves
    assert "החל מהשעה 15:00 עד השעה 20:30" in arrival.text
    assert "הגעה מוקדמת" in arrival.text
    assert "יש לפנות את האוהלים" in departure.text
    assert "לסיום שעות הפעילות בשעה 17:00" in departure.text


def test_the_cut_falls_on_the_topic_not_the_midpoint(sections):
    """A midpoint cut severs the 50% fee and the 17:00 limit from their context."""
    arrival, departure = [s for s in sections if "שעות כניסה ויציאה" in s.title]
    assert len(arrival.text) < len(departure.text)
    # The late-checkout rule and both its numbers stay in one chunk.
    assert "17:00" in departure.text
    assert "50%" in departure.text
    # Nothing about departure leaks into the arrival half.
    assert "עזיבה" not in arrival.text


def test_both_halves_keep_the_section_title_and_source(sections):
    halves = [s for s in sections if "שעות כניסה ויציאה" in s.title]
    assert {s.title for s in halves} == {"שעות כניסה ויציאה"}
    assert {s.source_url for s in halves} == {"https://www.parks.org.il/camping/x/"}


def test_no_text_is_lost_by_the_split(sections):
    arrival, departure = [s for s in sections if "שעות כניסה ויציאה" in s.title]
    assert not set(arrival.text.split("\n")) & set(departure.text.split("\n"))


def test_other_sections_are_not_split(sections):
    for title in ("מה בחניון?", "כניסת כלבים", "הערות למחירון"):
        assert len([s for s in sections if title in s.title]) == 1


def test_a_section_with_no_departure_line_is_left_whole():
    from source.scraper.rules_ingest.sections import Section, _split_topics

    only_arrival = Section("שעות כניסה ויציאה", "הכניסה החל מהשעה 15:00.")
    assert _split_topics(only_arrival) == [only_arrival]
