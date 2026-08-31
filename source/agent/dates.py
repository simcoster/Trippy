"""Date intent → stay windows. Calendar math lives here, not in the extract LLM."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.tools import StructuredTool

TZ_IL = ZoneInfo("Asia/Jerusalem")

MAX_DATE_WINDOWS = 4
DATE_TRUNCATED_NOTICE = (
    "יש יותר מ-4 טווחי תאריכים מתאימים; חיפשתי רק את ארבעת הראשונים."
)

_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_WEEKDAY_EN = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def today_il(today: date | None = None) -> date:
    if today is not None:
        return today
    return datetime.now(TZ_IL).date()


def iso_monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def weekday_this_iso_week(today: date, weekday: int) -> date:
    return iso_monday(today) + timedelta(days=int(weekday))


def weekday_next_iso_week(today: date, weekday: int) -> date:
    """Thursday of next ISO week — 10 days from Monday, never this week's 3."""
    return iso_monday(today) + timedelta(days=7 + int(weekday))


def next_friday(today: date | None = None) -> date:
    """Upcoming Friday; if today is Friday, the Friday one week later."""
    today = today_il(today)
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def next_saturday(today: date | None = None) -> date:
    return next_friday(today) + timedelta(days=1)


def _parse_iso_day(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value).strip()[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _as_stay_range(start: date, end: date | None = None) -> dict[str, str]:
    """Check-in / check-out ISO range. End is exclusive; always at least one night."""
    if end is None or end <= start:
        end = start + timedelta(days=1)
    return {"start": start.isoformat(), "end": end.isoformat()}


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _weekday_index(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int) and 0 <= value <= 6:
        return value
    text = str(value).strip().lower()
    if text.isdigit():
        n = int(text)
        return n if 0 <= n <= 6 else None
    return _WEEKDAY_INDEX.get(text)


def _nights_for(kind: str | None, nights: int | None) -> int:
    if nights is not None and nights > 0:
        return nights
    if kind == "weekend":
        return 2
    return 1


def _stay(start: date, nights: int) -> dict[str, str]:
    return _as_stay_range(start, start + timedelta(days=nights))


def _check_in_for_weekday(
    today: date,
    weekday: int,
    *,
    when: str | None,
    weeks_from_now: int | None,
) -> date | None:
    if weeks_from_now is not None:
        start = weekday_this_iso_week(today, weekday) + timedelta(
            days=7 * weeks_from_now
        )
        return start if start >= today else None
    ref = (when or "next").strip().lower()
    if ref == "this":
        start = weekday_this_iso_week(today, weekday)
        return start if start >= today else None
    return weekday_next_iso_week(today, weekday)


def _friday_for_weekend(
    today: date,
    *,
    when: str | None,
    weeks_from_now: int | None,
) -> date | None:
    return _check_in_for_weekday(
        today,
        4,
        when=when if when is not None else "this",
        weeks_from_now=weeks_from_now,
    )


def _enumerate_weekdays(
    today: date,
    weekday: int,
    *,
    horizon_days: int,
    nights: int,
) -> list[dict[str, str]]:
    last = today + timedelta(days=horizon_days)
    days = (weekday - today.weekday()) % 7
    cursor = today + timedelta(days=days)
    windows: list[dict[str, str]] = []
    while cursor < last:
        if cursor >= today:
            windows.append(_stay(cursor, nights))
        cursor += timedelta(days=7)
    return windows


def resolve_dates(
    nights: int | None = None,
    kind: str | None = None,
    weekday: str | int | None = None,
    when: str | None = None,
    weeks_from_now: int | None = None,
    horizon_days: int | None = None,
    on: str | None = None,
    start: str | None = None,
    end: str | None = None,
    today: date | str | None = None,
    **_extra: Any,
) -> dict[str, Any]:
    """Turn a date intent into at most MAX_DATE_WINDOWS stay ranges."""
    today_d = today_il(today if isinstance(today, date) else _parse_iso_day(today))
    kind_n = str(kind or "").strip().lower() or None
    when_n = str(when).strip().lower() if when else None
    nights_n = _as_int(nights)
    weeks_n = _as_int(weeks_from_now)
    horizon_n = _as_int(horizon_days)
    wd = _weekday_index(weekday)

    if kind_n in {"day", "date"}:
        kind_n = "on" if kind_n == "date" else "weekday"
    if kind_n is None:
        if wd is not None:
            kind_n = "weekday"
        elif on or start:
            kind_n = "on"
        elif horizon_n:
            kind_n = "weekend"

    stay_nights = _nights_for(kind_n, nights_n)
    windows: list[dict[str, str]] = []

    start_d = _parse_iso_day(start)
    end_d = _parse_iso_day(end)
    if start_d is not None:
        windows.append(_as_stay_range(start_d, end_d))
    elif on:
        on_d = (
            today_d
            if str(on).strip().lower() in {"today", "tonight"}
            else _parse_iso_day(on)
        )
        if on_d is None:
            on_d = today_d
        if weeks_n is not None:
            on_d = on_d + timedelta(weeks=weeks_n)
        windows.append(_stay(on_d, stay_nights))
    elif kind_n == "weekend":
        if horizon_n:
            windows.extend(
                _enumerate_weekdays(
                    today_d, 4, horizon_days=horizon_n, nights=stay_nights
                )
            )
        else:
            fri = _friday_for_weekend(
                today_d, when=when_n, weeks_from_now=weeks_n
            )
            if fri is None:
                fri = weekday_next_iso_week(today_d, 4)
            windows.append(_stay(fri, stay_nights))
    elif kind_n == "weekday" and wd is not None:
        if horizon_n:
            windows.extend(
                _enumerate_weekdays(
                    today_d, wd, horizon_days=horizon_n, nights=stay_nights
                )
            )
        else:
            check_in = _check_in_for_weekday(
                today_d, wd, when=when_n, weeks_from_now=weeks_n
            )
            if check_in is not None:
                windows.append(_stay(check_in, stay_nights))
    elif weeks_n is not None:
        windows.append(_stay(today_d + timedelta(weeks=weeks_n), stay_nights))

    truncated = len(windows) > MAX_DATE_WINDOWS
    windows = windows[:MAX_DATE_WINDOWS]
    return {
        "windows": windows,
        "truncated": truncated,
        "notice": DATE_TRUNCATED_NOTICE if truncated else None,
    }


def _resolve_dates_tool(
    nights: int | None = None,
    kind: str | None = None,
    weekday: str | None = None,
    when: str | None = None,
    weeks_from_now: int | None = None,
    horizon_days: int | None = None,
    on: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Map a date intent to ISO stay windows (max 4). Do not invent amenities."""
    return resolve_dates(
        nights=nights,
        kind=kind,
        weekday=weekday,
        when=when,
        weeks_from_now=weeks_from_now,
        horizon_days=horizon_days,
        on=on,
        start=start,
        end=end,
    )


resolve_dates_tool = StructuredTool.from_function(
    func=_resolve_dates_tool,
    name="resolve_dates",
    description=(
        "Resolve a date intent into ISO check-in/check-out windows. "
        "kind=weekday|weekend|on; when=this|next; weeks_from_now for "
        "'in N weeks'; horizon_days for sparse ranges (capped at 4). "
        "next weekday is next ISO week, not this week's upcoming day."
    ),
)


_RESOLVE_DATES_KEYS = (
    "nights",
    "kind",
    "weekday",
    "when",
    "weeks_from_now",
    "horizon_days",
    "on",
    "start",
    "end",
)


def intent_tool_args(intent: dict[str, Any]) -> dict[str, Any]:
    """Subset of a date intent that resolve_dates_tool accepts."""
    out: dict[str, Any] = {}
    for key in _RESOLVE_DATES_KEYS:
        value = intent.get(key)
        if value is None or value == "":
            continue
        if key == "weekday" and isinstance(value, int) and 0 <= value <= 6:
            value = _WEEKDAY_EN[value]
        out[key] = value
    return out


def apply_resolved_dates(
    constraints: dict[str, Any],
    resolved: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach windows / truncation notice onto a constraints dict."""
    windows = list((resolved or {}).get("windows") or [])
    truncated = bool((resolved or {}).get("truncated"))
    notice = (resolved or {}).get("notice")
    if len(windows) > MAX_DATE_WINDOWS:
        truncated = True
        notice = notice or DATE_TRUNCATED_NOTICE
        windows = windows[:MAX_DATE_WINDOWS]
    constraints["date_windows"] = windows
    constraints["date"] = windows[0] if windows else None
    constraints["date_truncated"] = truncated
    constraints["date_notice"] = notice
    return constraints
