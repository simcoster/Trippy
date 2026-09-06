"""Price-scrape lodging names connect to availability booking types."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

_SCRAPER_DIR = Path(__file__).resolve().parents[2] / "scraper"
sys.path.insert(0, str(_SCRAPER_DIR))

from info_site.classify import RateCardClassifier, classify_row  # noqa: E402
from info_site.match_listing import (  # noqa: E402
    InfoWebsiteNameMatcher,
    match_info_website_name,
)
from info_site.schemas import RawPriceRow  # noqa: E402

# Rate-card lodging product (info_website_names.name after classify).
TENT_LISTING = "לינת שטח באוהלים פרטיים"
# Guest/period live on list_prices, not in the listing name.
TENT_CHILD_WEEKEND_LABEL = "לינת שטח באוהלים פרטיים - ילד סוף שבוע"
# Booking engine name: same product, compound suffix — not an exact string match.
NORTHERN_COMPOUND_BOOKING = "לינת שטח באוהלים פרטיים - חניון צפוני"
LISTING_ID = 7
BOOKING_TYPE_ID = 99


def _classified_tent_child_weekend(classifier: RateCardClassifier):
    return classify_row(
        RawPriceRow(raw_label=TENT_CHILD_WEEKEND_LABEL, price=58.0),
        classifier=classifier,
    )


@pytest.mark.llm
def test_llm_tent_booking_matches_classified_rate_card_name():
    if not os.environ.get("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY required")
    payload = RateCardClassifier().classify_label(TENT_CHILD_WEEKEND_LABEL)
    assert payload.kind == "lodging"
    assert payload.guest_type == "child"
    assert payload.rate_period == "weekend_holiday"
    listings = [
        (LISTING_ID, payload.accommodation_type),
        (11, "בונגלו עם מזגן"),
    ]
    listing_id, confidence = match_info_website_name(
        NORTHERN_COMPOUND_BOOKING,
        listings,
        matcher=InfoWebsiteNameMatcher(),
    )
    assert listing_id == LISTING_ID
    assert confidence is None or 0.0 <= confidence <= 1.0
    # Without a matcher there is no model to ask, so nothing is matched.
    assert match_info_website_name(NORTHERN_COMPOUND_BOOKING, listings) == (None, None)
