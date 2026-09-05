"""The rate-card notes are parsed but not extracted (PLAN 2026-09-05).

Every note is per-rate by construction and the extractor drops the rate label,
so `child_min_age 5` and `weekend_min_nights 2` were stored as facts about the
campsite. Parked at the ingest step, not the parser: the parser's job is to
return what the page says, and its tests keep asserting the section is there.
"""

from __future__ import annotations

from source.scraper.rules_ingest.ingest import (
    PARKED_SECTION_TITLES,
    sections_to_extract,
)
from source.scraper.rules_ingest.sections import Section


def test_rate_notes_are_parked():
    assert "הערות למחירון" in PARKED_SECTION_TITLES


def test_parked_sections_are_removed_and_the_rest_keep_their_order():
    sections = [
        Section("מה בחניון?", "תאורת שטח"),
        Section("הערות למחירון", "בונגלו עם מזגן סופי שבוע וחגים: מותנה במינימום 2 לילות"),
        Section("כניסת כלבים", "הכניסה לכלבים אסורה"),
    ]
    kept = sections_to_extract(sections)
    assert [s.title for s in kept] == ["מה בחניון?", "כניסת כלבים"]
    assert kept[0] is sections[0] and kept[1] is sections[2]


def test_nothing_else_is_touched():
    sections = [Section("מה בחניון?", "x"), Section("נגישות", "y")]
    assert sections_to_extract(sections) == sections
