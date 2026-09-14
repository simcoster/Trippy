"""Availability scrape: page hashes, skip, and the change-report markdown."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.availability_report import (
    AvailabilityRun,
    HttpError,
    LayoutSuspicion,
    NightCountKey,
    VacancyChange,
    diff_night_counts,
    html_sha256,
    layout_suspicion,
    offers_sha256,
    render_run_report,
    should_skip_write,
    write_run_report,
)
from source.scraper.populate_availability import aggregate_offerings, parse_rooms


def test_offers_sha256_is_order_independent():
    a = [{"room_type": "בונגלו", "room_count": 2}, {"room_type": "חושה", "room_count": 1}]
    b = [{"room_type": "חושה", "room_count": 1}, {"room_type": "בונגלו", "room_count": 2}]
    assert offers_sha256(a) == offers_sha256(b)


def test_offers_sha256_changes_when_count_changes():
    a = [{"room_type": "בונגלו", "room_count": 2}]
    b = [{"room_type": "בונגלו", "room_count": 1}]
    assert offers_sha256(a) != offers_sha256(b)


def test_offers_sha256_stable_after_aggregate_reorder():
    offerings = [
        {"room_type": "בונגלו עם מזגן מספר 2"},
        {"room_type": "חושה מספר 1"},
        {"room_type": "בונגלו עם מזגן מספר 1"},
    ]
    flipped = list(reversed(offerings))
    assert offers_sha256(aggregate_offerings(offerings)) == offers_sha256(
        aggregate_offerings(flipped)
    )


def test_html_sha256_changes_when_html_changes():
    assert html_sha256("<html>a</html>") != html_sha256("<html>b</html>")
    assert html_sha256("<html>a</html>") == html_sha256("<html>a</html>")


def _be_results_html(*, viewstate: str, room_types: tuple[str, ...]) -> str:
    """Minimal INPA results page: roomData buttons plus ASP.NET chrome."""
    buttons: list[str] = []
    for index, name in enumerate(room_types):
        payload = json.dumps(
            {
                "RoomType": name,
                "Price": 100,
                "Currency": "₪",
                "RoomCode": str(index),
                "MatrixCode": "A",
            },
            ensure_ascii=False,
        )
        encoded = payload.replace('"', "&quot;")
        buttons.append(f'<button roomData="{encoded}"></button>')
    chrome = f'<input type="hidden" name="__VIEWSTATE" value="{viewstate}" />'
    return "<html>" + chrome + "".join(buttons) + "</html>"


def test_skip_uses_offers_hash_not_raw_html():
    """ViewState noise must not force a rewrite; a gone room must."""
    same_rooms = ("בונגלו עם מזגן", "חושה")
    yesterday = _be_results_html(viewstate="AAA", room_types=same_rooms)
    today_chrome = _be_results_html(viewstate="BBB", room_types=same_rooms)
    today_gone = _be_results_html(viewstate="BBB", room_types=("בונגלו עם מזגן",))

    stored_offers = offers_sha256(aggregate_offerings(parse_rooms(yesterday)))
    chrome_offers = offers_sha256(aggregate_offerings(parse_rooms(today_chrome)))
    gone_offers = offers_sha256(aggregate_offerings(parse_rooms(today_gone)))

    assert html_sha256(yesterday) != html_sha256(today_chrome)
    assert stored_offers == chrome_offers
    assert should_skip_write(stored_offers, chrome_offers) is True
    assert should_skip_write(stored_offers, gone_offers) is False


def test_should_skip_write_requires_a_stored_digest():
    digest = offers_sha256([{"room_type": "בונגלו", "room_count": 1}])
    assert should_skip_write(None, digest) is False
    assert should_skip_write(digest, digest) is True
    other = offers_sha256([{"room_type": "בונגלו", "room_count": 2}])
    assert should_skip_write(digest, other) is False


def test_layout_suspicion_empty_vs_previous_offers():
    had_rooms = offers_sha256([{"room_type": "בונגלו", "room_count": 1}])
    assert layout_suspicion(previous_offers_sha=had_rooms, new_aggregated=[]) is True
    assert (
        layout_suspicion(
            previous_offers_sha=offers_sha256([]),
            new_aggregated=[{"room_type": "בונגלו", "room_count": 1}],
        )
        is True
    )
    assert layout_suspicion(previous_offers_sha=None, new_aggregated=[]) is False
    assert (
        layout_suspicion(previous_offers_sha=offers_sha256([]), new_aggregated=[])
        is False
    )


def test_diff_night_counts_appeared_gone_and_count_change():
    bungalow = NightCountKey(site_id=2, type_name="בונגלו")
    hut = NightCountKey(site_id=2, type_name="חושה")
    tent = NightCountKey(site_id=2, type_name="אוהל")
    old = {bungalow: 3, hut: 1}
    new = {bungalow: 1, tent: 2}
    changes = diff_night_counts(
        site_name="Achziv", start=date(2026, 9, 20), old=old, new=new
    )
    by_type = {c.type_name: c for c in changes}
    assert by_type["בונגלו"] == VacancyChange(
        site_id=2,
        site_name="Achziv",
        start_date=date(2026, 9, 20),
        type_name="בונגלו",
        old_count=3,
        new_count=1,
    )
    assert by_type["חושה"].old_count == 1 and by_type["חושה"].new_count is None
    assert by_type["אוהל"].old_count is None and by_type["אוהל"].new_count == 2
    assert "אוהל" in by_type and "חושה" in by_type
    unchanged = diff_night_counts(
        site_name="Achziv", start=date(2026, 9, 20), old=old, new=old
    )
    assert unchanged == []


def test_render_run_report_lists_only_changes(tmp_path: Path):
    run = AvailabilityRun(
        started_at=datetime(2026, 9, 14, 5, 0, tzinfo=timezone.utc),
        seconds=12.3,
        sites=2,
        nights=14,
        pages_fetched=28,
        pages_skipped=20,
        rows_upserted=8,
        changes=[
            VacancyChange(
                site_id=2,
                site_name="Achziv",
                start_date=date(2026, 9, 20),
                type_name="בונגלו",
                old_count=3,
                new_count=1,
            )
        ],
        unmatched=[(2, "עמדה לקרוואן")],
        http_errors=[
            HttpError(
                site_id=3,
                site_name="Masada",
                start_date=date(2026, 9, 21),
                message="503",
            )
        ],
        layout_suspicions=[
            LayoutSuspicion(
                site_id=2,
                site_name="Achziv",
                start_date=date(2026, 9, 22),
                previous_had_offers=True,
                now_empty=True,
            )
        ],
    )
    text = render_run_report(run, LlmUsage())
    assert "3 → 1" in text
    assert "skipped: 20" in text
    assert "עמדה לקרוואן" in text
    assert "503" in text
    assert "parser returned none" in text
    dest = tmp_path / "report.md"
    written = write_run_report(text, path=dest)
    assert written == dest
    assert dest.read_text(encoding="utf-8") == text
