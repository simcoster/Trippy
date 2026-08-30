"""Normalize INPA room names: strip unit numbers so listings collapse to one type."""

from __future__ import annotations

import sys
from pathlib import Path

# populate_availability imports amenity_enrichment from source/scraper cwd.
_SCRAPER_DIR = Path(__file__).resolve().parents[2] / "scraper"
sys.path.insert(0, str(_SCRAPER_DIR))

from populate_availability import (  # noqa: E402
    aggregate_offerings,
    normalize_accommodation_name,
)


def test_normalize_strips_single_unit_number():
    assert normalize_accommodation_name("בונגלו עם מזגן מספר 1") == "בונגלו עם מזגן"
    assert normalize_accommodation_name("בונגלו עם מזגן מספר 42") == "בונגלו עם מזגן"


def test_normalize_strips_unit_number_range():
    assert normalize_accommodation_name("חושה כפולה מספר 1-4") == "חושה כפולה"
    assert normalize_accommodation_name("חושה כפולה מספר 7-8") == "חושה כפולה"
    assert normalize_accommodation_name("חושה כפולה מספר 1 - 4") == "חושה כפולה"


def test_normalize_strips_trailing_unit_number_without_mispar():
    assert normalize_accommodation_name("בונגלו עם מזגן 01") == "בונגלו עם מזגן"
    assert normalize_accommodation_name("בונגלו עם מזגן 15") == "בונגלו עם מזגן"
    assert normalize_accommodation_name("עמדה לקרוואן פרטי חניה 05") == (
        "עמדה לקרוואן פרטי חניה"
    )
    assert normalize_accommodation_name("חדר צוות מאובזר 02") == "חדר צוות מאובזר"


def test_normalize_collapses_spaces_and_keeps_accessible_distinct():
    assert normalize_accommodation_name("בונגלו  עם מזגן 11") == "בונגלו עם מזגן"
    assert normalize_accommodation_name("בונגלו עם מזגן  15") == "בונגלו עם מזגן"
    assert normalize_accommodation_name("בונגלו מונגש עם מזגן 17") == (
        "בונגלו מונגש עם מזגן"
    )


def test_normalize_leaves_name_without_suffix():
    assert normalize_accommodation_name("אוהל משפחתי") == "אוהל משפחתי"
    assert normalize_accommodation_name("לינת שטח באוהלים פרטיים") == (
        "לינת שטח באוהלים פרטיים"
    )
    assert normalize_accommodation_name("חדר צוות עץ") == "חדר צוות עץ"


def test_normalize_empty():
    assert normalize_accommodation_name("") == ""
    assert normalize_accommodation_name(None) == ""  # type: ignore[arg-type]


def test_aggregate_merges_range_and_single_suffix():
    offerings = [
        {"room_type": "חושה כפולה מספר 1-4", "price": 400},
        {"room_type": "חושה כפולה מספר 7-8", "price": 380},
        {"room_type": "חושה כפולה מספר 9", "price": 420},
    ]
    grouped = aggregate_offerings(offerings)
    assert len(grouped) == 1
    assert grouped[0]["room_type"] == "חושה כפולה"
    assert grouped[0]["room_count"] == 3
    assert grouped[0]["price"] == 380


def test_aggregate_merges_trailing_padded_unit_numbers():
    offerings = [
        {"room_type": "בונגלו עם מזגן 01", "price": 430},
        {"room_type": "בונגלו  עם מזגן 11", "price": 430},
        {"room_type": "בונגלו עם מזגן  15", "price": 430},
        {"room_type": "בונגלו מונגש עם מזגן 17", "price": 430},
    ]
    grouped = {row["room_type"]: row for row in aggregate_offerings(offerings)}
    assert set(grouped) == {"בונגלו עם מזגן", "בונגלו מונגש עם מזגן"}
    assert grouped["בונגלו עם מזגן"]["room_count"] == 3
    assert grouped["בונגלו מונגש עם מזגן"]["room_count"] == 1
