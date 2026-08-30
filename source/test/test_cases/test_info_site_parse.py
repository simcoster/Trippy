"""Parse parks.org.il info-page rate tables and booking iframe."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRAPER_DIR = Path(__file__).resolve().parents[2] / "scraper"
sys.path.insert(0, str(_SCRAPER_DIR))

from info_site.newsflashes import parse_flashbacks_json  # noqa: E402
from info_site.parse import (  # noqa: E402
    parse_booking_hotel_id,
    parse_price,
    parse_rate_table,
    parse_whats_new,
    parse_wp_post_id,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "info_site"
_TABLE_HTML = (_FIXTURES / "horashat_tal_table.html").read_text(encoding="utf-8")
_WHATS_NEW_HTML = (_FIXTURES / "listing_whats_new.html").read_text(encoding="utf-8")


def test_parse_price_shekel():
    assert parse_price("76.00 ₪") == 76.0
    assert parse_price("265.00 ₪") == 265.0
    assert parse_price("no price") is None


def test_parse_horashat_tal_table1_has_fifteen_rows():
    rows = parse_rate_table(_TABLE_HTML)
    assert len(rows) == 15
    labels = [row["raw_label"] for row in rows]
    assert "לינת שטח באוהלים פרטיים - מבוגר" in labels
    assert "לינה בבונגלו עם מזגן אמצע שבוע" in labels
    adult = next(row for row in rows if row["raw_label"].endswith("מבוגר") and "קרוואן" not in row["raw_label"])
    assert adult["price"] == 76.0
    assert adult["notes"] == "גיל 14 ומעלה"


def test_parse_booking_hotel_id_and_wp_post_id():
    assert parse_booking_hotel_id(_TABLE_HTML) == "9_1"
    assert parse_wp_post_id(_TABLE_HTML) == "14874"


def test_parse_flashbacks_json_fixture():
    payload = [
        {
            "title": "שינוי בתנאי הביטול ביחידות האירוח",
            "permalink": "https://www.parks.org.il/newsflash/cancellation-change/",
        }
    ]
    items = parse_flashbacks_json(payload)
    assert len(items) == 1
    assert items[0]["title"].startswith("שינוי")
    assert "flashback" in items[0]["html"]


def test_parse_whats_new_listing_bullets():
    items = parse_whats_new(_WHATS_NEW_HTML)
    assert len(items) == 4
    assert any("מעיין חרוד" in item for item in items)
