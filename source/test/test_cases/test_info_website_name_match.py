"""Exact then LLM match from booking names to info_website_names."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_SCRAPER_DIR = Path(__file__).resolve().parents[2] / "scraper"
sys.path.insert(0, str(_SCRAPER_DIR))

from info_site.match_listing import (  # noqa: E402
    InfoWebsiteNameMatcher,
    match_info_website_name,
)
from populate_availability import (  # noqa: E402
    ensure_booking_accommodation_type,
    normalize_accommodation_name,
)

LISTINGS = [
    (7, "לינת שטח באוהלים פרטיים"),
    (11, "בונגלו עם מזגן"),
    (20, "עמדת חניה לקרוואן"),
]


def test_exact_booking_name_matches_info_site_name():
    needle = normalize_accommodation_name("בונגלו עם מזגן מספר 1")
    assert match_info_website_name(needle, LISTINGS) == 11


def test_mismatch_calls_llm_and_returns_picked_listing():
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    matcher.pick_name.return_value = "עמדת חניה לקרוואן"
    listing_id = match_info_website_name(
        "עמדה לקרוואן פרטי חניה",
        LISTINGS,
        matcher=matcher,
    )
    assert listing_id == 20
    matcher.pick_name.assert_called_once()
    assert matcher.pick_name.call_args.args[0] == "עמדה לקרוואן פרטי חניה"
    assert matcher.pick_name.call_args.args[1] == [name for _, name in LISTINGS]


def test_exact_match_does_not_call_llm():
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    listing_id = match_info_website_name("בונגלו עם מזגן", LISTINGS, matcher=matcher)
    assert listing_id == 11
    matcher.pick_name.assert_not_called()


def test_empty_listings_returns_none_without_llm():
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    assert match_info_website_name("בונגלו עם מזגן", [], matcher=matcher) is None
    matcher.pick_name.assert_not_called()


def test_llm_null_is_unmatched():
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    matcher.pick_name.return_value = None
    assert match_info_website_name("חושה כפולה", LISTINGS, matcher=matcher) is None


def test_already_linked_type_does_not_call_llm():
    cur = MagicMock()
    cur.fetchone.return_value = (42, 7)
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    type_id = ensure_booking_accommodation_type(
        cur,
        hotel_id=1,
        name="בונגלו עם מזגן",
        listings=LISTINGS,
        matcher=matcher,
    )
    assert type_id == 42
    matcher.pick_name.assert_not_called()
    executed = [call.args[0] for call in cur.execute.call_args_list]
    assert not any("info_website_name_id = %(info_website_name_id)s" in sql for sql in executed)


def test_matcher_rejects_name_not_on_list():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"name": "invented"}')
            )
        ],
    )
    matcher = InfoWebsiteNameMatcher(client=client)
    assert matcher.pick_name("בונגלו", ["עמדת חניה לקרוואן"]) is None
