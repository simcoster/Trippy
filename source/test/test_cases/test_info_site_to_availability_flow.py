"""Price-scrape lodging names connect to availability booking types."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

load_dotenv()

_SCRAPER_DIR = Path(__file__).resolve().parents[2] / "scraper"
sys.path.insert(0, str(_SCRAPER_DIR))

from info_site.classify import RateCardClassifier, classify_row, lodging_rows_to_persist  # noqa: E402
from info_site.db import snapshot_list_prices  # noqa: E402
from info_site.match_listing import (  # noqa: E402
    InfoWebsiteNameMatcher,
    match_info_website_name,
)
from info_site.schemas import ClassificationPayload, RawPriceRow  # noqa: E402
from populate_availability import ensure_booking_accommodation_type  # noqa: E402

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


def test_tent_child_weekend_rate_connects_to_northern_compound_booking():
    """Classified tents+child+weekend share info_website_name_id with the compound booking."""
    classifier = MagicMock(spec=RateCardClassifier)
    classifier.classify_label.return_value = ClassificationPayload(
        accommodation_type=TENT_LISTING,
        guest_type="child",
        rate_period="weekend_holiday",
        kind="lodging",
    )
    classified = _classified_tent_child_weekend(classifier)
    assert classified.accommodation_type == TENT_LISTING
    assert classified.guest_type == "child"
    assert classified.rate_period == "weekend_holiday"
    assert lodging_rows_to_persist([classified]) == [classified]

    price_cur = MagicMock()
    price_cur.fetchone.return_value = (LISTING_ID,)
    price_ctx = MagicMock()
    price_ctx.__enter__.return_value = price_cur
    conn = MagicMock()
    conn.cursor.return_value = price_ctx
    snapshot_list_prices(conn, site_id=3, rows=[classified])
    insert = next(
        call
        for call in price_cur.execute.call_args_list
        if "INSERT INTO list_prices" in call.args[0]
    )
    assert insert.args[1]["info_website_name_id"] == LISTING_ID
    assert insert.args[1]["guest_type"] == "child"
    assert insert.args[1]["rate_period"] == "weekend_holiday"
    assert insert.args[1]["raw_label"] == TENT_CHILD_WEEKEND_LABEL

    listings = [(LISTING_ID, TENT_LISTING), (11, "בונגלו עם מזגן")]
    assert match_info_website_name(NORTHERN_COMPOUND_BOOKING, listings) is None

    matcher = MagicMock(spec=InfoWebsiteNameMatcher)
    matcher.pick_name.return_value = TENT_LISTING
    booking_cur = MagicMock()
    booking_cur.fetchone.return_value = (BOOKING_TYPE_ID, None)
    type_id = ensure_booking_accommodation_type(
        booking_cur,
        hotel_id=1,
        name=NORTHERN_COMPOUND_BOOKING,
        listings=listings,
        matcher=matcher,
    )
    assert type_id == BOOKING_TYPE_ID
    matcher.pick_name.assert_called_once()
    link = next(
        call
        for call in booking_cur.execute.call_args_list
        if "info_website_name_id = %(info_website_name_id)s" in call.args[0]
    )
    assert link.args[1] == {
        "id": BOOKING_TYPE_ID,
        "info_website_name_id": LISTING_ID,
    }


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
    listing_id = match_info_website_name(
        NORTHERN_COMPOUND_BOOKING,
        listings,
        matcher=InfoWebsiteNameMatcher(),
    )
    assert listing_id == LISTING_ID
    assert match_info_website_name(NORTHERN_COMPOUND_BOOKING, listings) is None
