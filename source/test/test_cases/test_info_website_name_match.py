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
    normalize_accommodation_name,
)

LISTINGS = [
    (7, "לינת שטח באוהלים פרטיים"),
    (11, "בונגלו עם מזגן"),
    (20, "עמדת חניה לקרוואן"),
]


def test_exact_booking_name_matches_info_site_name():
    """An exact hit carries no confidence: there is nothing to be unsure of."""
    needle = normalize_accommodation_name("בונגלו עם מזגן מספר 1")
    assert match_info_website_name(needle, LISTINGS) == (11, None)


def test_mismatch_calls_llm_and_returns_picked_listing():
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    matcher.pick_name.return_value = ("עמדת חניה לקרוואן", 0.82)
    listing_id, confidence = match_info_website_name(
        "עמדה לקרוואן פרטי חניה",
        LISTINGS,
        matcher=matcher,
    )
    assert (listing_id, confidence) == (20, 0.82)
    matcher.pick_name.assert_called_once()
    assert matcher.pick_name.call_args.args[0] == "עמדה לקרוואן פרטי חניה"
    assert matcher.pick_name.call_args.args[1] == [name for _, name in LISTINGS]


def test_exact_match_does_not_call_llm():
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    listing_id, _ = match_info_website_name("בונגלו עם מזגן", LISTINGS, matcher=matcher)
    assert listing_id == 11
    matcher.pick_name.assert_not_called()


def test_empty_listings_returns_none_without_llm():
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    assert match_info_website_name("בונגלו עם מזגן", [], matcher=matcher) == (
        None,
        None,
    )
    matcher.pick_name.assert_not_called()


def test_a_refusal_is_passed_up_with_its_confidence():
    """The listing prompt forbids a null -- a price belongs somewhere -- so this
    is the model failing to follow it. `snapshot_list_prices` forces a match at
    confidence 0 rather than dropping the price, and says so in the run."""
    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    matcher.pick_name.return_value = (None, 0.1)
    assert match_info_website_name("חושה כפולה", LISTINGS, matcher=matcher) == (
        None,
        0.1,
    )


def test_matcher_rejects_name_not_on_list():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"name": "invented", "confidence": 0.9}'
                )
            )
        ],
    )
    matcher = InfoWebsiteNameMatcher(client=client)
    # Confidence survives; the off-list name does not, however sure the model is.
    assert matcher.pick_name("בונגלו", ["עמדת חניה לקרוואן"]) == (None, 0.9)
