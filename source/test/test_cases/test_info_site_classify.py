"""Classify info-site rate-card rows: fee heuristic + optional 30B LLM."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

_SCRAPER_DIR = Path(__file__).resolve().parents[2] / "scraper"
sys.path.insert(0, str(_SCRAPER_DIR))

from info_site.classify import (  # noqa: E402
    RateCardClassifier,
    classify_fee_row,
    classify_row,
    is_fee_label,
    lodging_rows_to_persist,
)
from info_site.parse import parse_rate_table  # noqa: E402
from info_site.schemas import ClassifiedPriceRow, RawPriceRow  # noqa: E402

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "info_site"
_TABLE_HTML = (_FIXTURES / "horashat_tal_table.html").read_text(encoding="utf-8")


def _bungalow_late_checkout() -> dict:
    rows = parse_rate_table(_TABLE_HTML)
    return next(row for row in rows if row["price"] == 265.0)


def test_is_fee_label_tosefet():
    fee = _bungalow_late_checkout()
    assert is_fee_label(fee["raw_label"])
    assert not is_fee_label("לינה בבונגלו עם מזגן אמצע שבוע")


def test_late_checkout_fee_classified_without_llm():
    raw = _bungalow_late_checkout()
    classified = classify_row(raw)
    assert classified.kind == "fee"
    assert classified.price == 265.0
    assert "יציאה מאוחרת" in raw["raw_label"] or "בונגלו" in classified.accommodation_type


def test_classify_fee_row_guest_and_period():
    row = classify_fee_row(
        RawPriceRow(raw_label="תוספת מבוגר בקרוואן", price=76.0, notes="גיל 14 ומעלה")
    )
    assert row.kind == "fee"
    assert row.guest_type == "adult"
    assert row.rate_period == "any"


def test_lodging_rows_to_persist_skips_fees():
    fee = classify_row(_bungalow_late_checkout())
    lodging = ClassifiedPriceRow(
        raw_label="לינה בבונגלו עם מזגן אמצע שבוע",
        price=430.0,
        accommodation_type="בונגלו עם מזגן",
        guest_type="any",
        rate_period="weekday",
        kind="lodging",
    )
    persisted = lodging_rows_to_persist([fee, lodging])
    assert persisted == [lodging]
    assert all(row.kind != "fee" for row in persisted)


@pytest.fixture(scope="module")
def classifier():
    if not (os.environ.get("NEBIUS_API_KEY") or os.environ.get("NEBULA_API_KEY")):
        pytest.skip("NEBIUS_API_KEY (or NEBULA_API_KEY) required")
    return RateCardClassifier()


def test_llm_classifies_bungalow_weekday(classifier):
    payload = classifier.classify_label("לינה בבונגלו עם מזגן אמצע שבוע")
    assert "בונגלו" in payload.accommodation_type
    assert payload.guest_type == "any"
    assert payload.rate_period == "weekday"
    assert payload.kind == "lodging"
