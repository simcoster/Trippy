"""Availability scraper: 1-adult search and booking-to-info-site name match."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

_SCRAPER_DIR = Path(__file__).resolve().parents[2] / "scraper"
sys.path.insert(0, str(_SCRAPER_DIR))

from info_site.match_listing import match_info_website_name  # noqa: E402
from populate_availability import (  # noqa: E402
    load_config,
    normalize_accommodation_name,
    search_url,
)


def test_config_and_search_url_use_one_adult():
    config = load_config()
    assert int(config["availability"]["adults"]) == 1
    url = search_url("9_1", date(2026, 9, 1), date(2026, 9, 2))
    assert "ad1=1" in url
    assert "ad1=2" not in url


def test_match_exact_normalized_name():
    """An exact hit carries no confidence: there is nothing to be unsure of."""
    listings = [(11, "בונגלו עם מזגן"), (12, "חדר צוות עץ")]
    needle = normalize_accommodation_name("בונגלו עם מזגן מספר 1")
    assert match_info_website_name(needle, listings) == (11, None)
    assert match_info_website_name("חדר צוות עץ", listings) == (12, None)


def test_without_a_matcher_there_is_no_match_to_make():
    """The model is what finds a near match, and it is told never to refuse --
    but with no matcher passed there is nothing to ask, so the caller decides."""
    listings = [(11, "בונגלו עם מזגן")]
    assert match_info_website_name("עמדה לקרוואן פרטי חניה", listings) == (None, None)
    assert match_info_website_name("חושה כפולה", listings) == (None, None)
