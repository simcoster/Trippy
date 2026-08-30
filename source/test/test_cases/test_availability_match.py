"""Availability scraper: 1-adult search and match-only accommodation types."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

_SCRAPER_DIR = Path(__file__).resolve().parents[2] / "scraper"
sys.path.insert(0, str(_SCRAPER_DIR))

from populate_availability import (  # noqa: E402
    UnknownAccommodationTypeError,
    load_config,
    match_accommodation_type,
    require_existing_accommodation_type,
    search_url,
)


def test_config_and_search_url_use_one_adult():
    config = load_config()
    assert int(config["availability"]["adults"]) == 1
    url = search_url("9_1", date(2026, 9, 1), date(2026, 9, 2))
    assert "ad1=1" in url
    assert "ad1=2" not in url


def test_match_exact_normalized_name():
    catalog = [(11, "בונגלו עם מזגן"), (12, "חדר צוות עץ")]
    assert match_accommodation_type("בונגלו עם מזגן מספר 1", catalog) == 11
    assert match_accommodation_type("חדר צוות עץ", catalog) == 12


def test_unknown_accommodation_type_raises():
    catalog = [(11, "בונגלו עם מזגן")]
    assert match_accommodation_type("חושה כפולה", catalog) is None
    with pytest.raises(UnknownAccommodationTypeError, match="חושה כפולה"):
        require_existing_accommodation_type(
            "חושה כפולה", catalog, hotel_id=1
        )


def test_unknown_type_does_not_invent_catalog_row():
    catalog = [(11, "בונגלו עם מזגן")]
    before = list(catalog)
    with pytest.raises(UnknownAccommodationTypeError):
        require_existing_accommodation_type("אוהל משפחתי", catalog, hotel_id=1)
    assert catalog == before
