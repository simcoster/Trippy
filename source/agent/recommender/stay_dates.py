"""Spoken date labels for one recommended stay."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any, NamedTuple

BOOKING_LINK_TEXT = "booking link"


class StayWindow(NamedTuple):
    start: str
    end: str
    booking_url: str


def day_month(iso: str) -> str:
    parts = (iso or "").strip().split("-")
    if len(parts) < 3:
        return (iso or "").strip()
    try:
        return f"{int(parts[2])}.{int(parts[1])}"
    except ValueError:
        return (iso or "").strip()


def _parse_day(iso: str) -> date | None:
    text = (iso or "").strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _window_label(window: StayWindow) -> str:
    start = day_month(window.start)
    end = day_month(window.end)
    if end and end != start:
        return f"{start}–{end}"
    return start


def _format_span(first: date, last: date, *, lead: bool) -> str:
    """First span in a month is `21–28.9`. A later span keeps both months: `11.10–15.10`."""
    if first == last:
        return day_month(first.isoformat())
    if lead and first.month == last.month and first.year == last.year:
        return f"{first.day}–{last.day}.{last.month}"
    return f"{day_month(first.isoformat())}–{day_month(last.isoformat())}"


def _check_in_spans(windows: Sequence[StayWindow]) -> list[str]:
    """Contiguous check-in days become one range. A gap starts another."""
    days: list[date] = []
    seen: set[date] = set()
    for window in windows:
        day = _parse_day(window.start)
        if day is None or day in seen:
            continue
        seen.add(day)
        days.append(day)
    days.sort()
    if not days:
        return []
    spans: list[str] = []
    run_start = days[0]
    previous = days[0]
    for day in days[1:]:
        if day - previous == timedelta(days=1):
            previous = day
            continue
        spans.append(_format_span(run_start, previous, lead=not spans))
        run_start = day
        previous = day
    spans.append(_format_span(run_start, previous, lead=not spans))
    return spans


def windows_from_dates(
    dates: Any, *, fallback: StayWindow | None = None
) -> tuple[StayWindow, ...]:
    """Night rows on a fit, or one fallback window when `dates` is empty."""
    windows: list[StayWindow] = []
    if isinstance(dates, list):
        for night in dates:
            if not isinstance(night, dict):
                continue
            start = str(night.get("start") or "").strip()
            end = str(night.get("end") or "").strip()
            if not start or not end:
                continue
            url = str(night.get("booking_url") or "").strip()
            windows.append(StayWindow(start, end, url))
    if windows:
        return tuple(windows)
    if fallback is not None and fallback.start and fallback.end:
        return (fallback,)
    return ()


def booking_lines(
    windows: Sequence[StayWindow],
    fallback_url: str,
) -> list[str]:
    """One link for the site."""
    url = (fallback_url or "").strip()
    if not url:
        url = next((window.booking_url for window in windows if window.booking_url), "")
    if not url:
        return []
    return [f"   [{BOOKING_LINK_TEXT}]({url})"]


def stay_date_label(windows: Sequence[StayWindow]) -> str:
    """One stay stays start–end. Several check-ins collapse to ranges."""
    cleaned = [window for window in windows if window.start and window.end]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return _window_label(cleaned[0])
    return ", ".join(_check_in_spans(cleaned))
