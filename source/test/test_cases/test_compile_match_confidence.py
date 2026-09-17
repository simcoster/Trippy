"""Compile keeps only rate rows that confidently match catalog lodging."""

from __future__ import annotations

from source.scraper.info_site.compile_price import match_compile_rows
from source.scraper.info_site.db import UNCERTAIN_BELOW
from source.scraper.info_site.parse import GatheredRateRow

NAMES = [(1, "חושה"), (2, "לינת שטח באוהלים פרטיים")]


def _row(tab: str, label: str, price: float, tab_id: str = "1") -> GatheredRateRow:
    return GatheredRateRow(
        tab_id=tab_id,
        rate_class=tab,
        raw_label=label,
        price=price,
        notes=None,
    )


def test_exact_and_confident_lodging_rows_are_kept(monkeypatch):
    gathered = [
        _row("רגיל", "חושה אמצע שבוע", 350.0),
        _row("מנוי", "לינת שטח באוהלים פרטיים - מבוגר", 57.0),
    ]

    def fake_resolve(needle, _names, **_kwargs):
        if needle.startswith("חושה"):
            return [1], None
        return [2], 0.95

    monkeypatch.setattr(
        "source.scraper.info_site.compile_price.resolve_listing_ids",
        fake_resolve,
    )
    rows = match_compile_rows(gathered, NAMES)
    assert [row.guest_type for row in rows] == ["רגיל", "מנוי"]
    assert [row.lodging for row in rows] == [
        "חושה",
        "לינת שטח באוהלים פרטיים",
    ]


def test_low_confidence_extras_tab_is_dropped(monkeypatch):
    gathered = [
        _row("רגיל", "חושה אמצע שבוע", 350.0),
        _row("ציוד להשכרה", "השכרת מזרן ללילה", 12.0, tab_id="9"),
        _row("ציוד להשכרה", "השכרת פלטה", 25.0, tab_id="9"),
    ]

    def fake_resolve(needle, _names, **_kwargs):
        if "השכרת" in needle:
            return [1], UNCERTAIN_BELOW - 0.1
        return [1], None

    monkeypatch.setattr(
        "source.scraper.info_site.compile_price.resolve_listing_ids",
        fake_resolve,
    )
    rows = match_compile_rows(gathered, NAMES)
    assert len(rows) == 1
    assert rows[0].guest_type == "רגיל"
    assert rows[0].label == "חושה אמצע שבוע"
    assert "ציוד להשכרה" not in {row.guest_type for row in rows}


def test_compile_does_not_force_a_refused_match(monkeypatch):
    gathered = [_row("ציוד להשכרה", "השכרת מזרן ללילה", 12.0, tab_id="9")]

    def fake_resolve(needle, names, **kwargs):
        assert kwargs.get("force") is False
        return [], 0.4

    monkeypatch.setattr(
        "source.scraper.info_site.compile_price.resolve_listing_ids",
        fake_resolve,
    )
    assert match_compile_rows(gathered, NAMES) == []
