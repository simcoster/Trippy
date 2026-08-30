"""Constraint schema helpers: dates and amenity OR groups."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

TZ_IL = ZoneInfo("Asia/Jerusalem")

_DATE_IN_QUERY_RE = re.compile(
    r"(?i)("
    r"next\s+friday|this\s+friday|coming\s+friday|"
    r"next\s+saturday|this\s+weekend|next\s+weekend|"
    r"ל?שישי\s+הבא|השישי\s+הבא|סופ.?״?ש\s+הבא|סופש\s+הבא"
    r")"
)


def today_il(today: date | None = None) -> date:
    if today is not None:
        return today
    return datetime.now(TZ_IL).date()


def next_friday(today: date | None = None) -> date:
    """Upcoming Friday; if today is Friday, the Friday one week later."""
    today = today_il(today)
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def next_saturday(today: date | None = None) -> date:
    fri = next_friday(today)
    return fri + timedelta(days=1)


def _parse_iso_day(value: Any) -> date | None:
    if value is None:
        return None
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


def resolve_relative_date_phrase(
    phrase: str,
    *,
    today: date | None = None,
) -> dict[str, str] | None:
    """Map a relative date phrase to {start, end} check-in/check-out ISO dates."""
    t = (phrase or "").strip().lower()
    if not t:
        return None
    today = today_il(today)

    if "weekend" in t or "סופ" in t:
        start = next_friday(today)
        # Fri night + Sat night → checkout Sunday
        return _as_stay_range(start, start + timedelta(days=2))

    if "friday" in t or "שישי" in t:
        return _as_stay_range(next_friday(today))

    if "saturday" in t or "שבת" in t:
        return _as_stay_range(next_saturday(today))

    return None


def _normalize_amenity_item(item: Any) -> str | dict[str, Any] | None:
    if item is None:
        return None
    if isinstance(item, str):
        text = item.strip()
        return text or None
    if isinstance(item, dict):
        op = str(item.get("op") or "and").strip().lower()
        values = item.get("values")
        if values is None and "query" in item:
            q = str(item.get("query") or "").strip()
            return q or None
        if not isinstance(values, list):
            return None
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        if not cleaned:
            return None
        if op == "or":
            return {"op": "or", "values": cleaned}
        # AND group of strings → flatten as separate AND terms later; keep as list expand
        if len(cleaned) == 1:
            return cleaned[0]
        return {"op": "or", "values": cleaned} if op == "or" else cleaned[0]
    return str(item).strip() or None


def _normalize_amenities(raw: Any) -> list[str | dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        item = _normalize_amenity_item(raw)
        return [item] if item is not None else []
    out: list[str | dict[str, Any]] = []
    for item in raw:
        norm = _normalize_amenity_item(item)
        if norm is None:
            continue
        out.append(norm)
    return out


def _date_from_semantic_queries(
    semantic: list[Any],
    *,
    today: date | None = None,
) -> tuple[dict[str, str] | None, list[Any]]:
    """Pull date intent out of semantic_constraints into a date range."""
    kept: list[Any] = []
    found: dict[str, str] | None = None
    for item in semantic or []:
        query = ""
        if isinstance(item, dict):
            query = str(item.get("query") or "")
        elif isinstance(item, str):
            query = item
        match = _DATE_IN_QUERY_RE.search(query)
        if match and found is None:
            found = resolve_relative_date_phrase(match.group(1), today=today)
            # Drop date-only semantic crumbs; keep if other content remains
            rest = _DATE_IN_QUERY_RE.sub("", query).strip(" ,.-")
            if rest and isinstance(item, dict):
                kept.append({**item, "query": rest})
            elif rest:
                kept.append({"query": rest} if not isinstance(item, dict) else item)
            continue
        kept.append(item)
    return found, kept


def _normalize_date_field(raw: Any, *, today: date | None = None) -> dict[str, str] | None:
    if raw is None or raw == "" or raw == []:
        return None
    if isinstance(raw, dict):
        # Relative phrase in start/end or "query"
        for key in ("query", "relative", "text"):
            if key in raw and isinstance(raw[key], str):
                resolved = resolve_relative_date_phrase(raw[key], today=today)
                if resolved:
                    return resolved
        start = _parse_iso_day(raw.get("start") or raw.get("from"))
        end = _parse_iso_day(raw.get("end") or raw.get("to") or raw.get("start") or raw.get("from"))
        if start and end:
            if end < start:
                start, end = end, start
            return _as_stay_range(start, end)
        # Single ISO under "date"
        single = _parse_iso_day(raw.get("date") or raw.get("day"))
        if single:
            return _as_stay_range(single)
        return None
    if isinstance(raw, list):
        days = [_parse_iso_day(x) for x in raw]
        days = [d for d in days if d is not None]
        if not days:
            # list of relative phrases
            for x in raw:
                resolved = resolve_relative_date_phrase(str(x), today=today)
                if resolved:
                    return resolved
            return None
        return _as_stay_range(min(days), max(days))
    if isinstance(raw, str):
        iso = _parse_iso_day(raw)
        if iso:
            return _as_stay_range(iso)
        return resolve_relative_date_phrase(raw, today=today)
    return None


def parse_constraints_dict(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse LLM JSON text into a dict (may be incomplete)."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def normalize_constraints(
    data: dict[str, Any] | str | None,
    *,
    today: date | None = None,
    user_text: str | None = None,
) -> dict[str, Any]:
    """
    Normalize extractor output to:
      date: {start, end} | null
      amenities: [str | {op: or, values: [...]}]
      numeric_constraints: [...]
      semantic_constraints: [{query: ...}, ...]
    """
    parsed = parse_constraints_dict(data)
    today = today_il(today)

    semantic = list(parsed.get("semantic_constraints") or [])
    numeric = list(parsed.get("numeric_constraints") or [])
    amenities = _normalize_amenities(parsed.get("amenities"))

    date_range = _normalize_date_field(parsed.get("date"), today=today)
    from_semantic, semantic = _date_from_semantic_queries(semantic, today=today)
    if date_range is None:
        date_range = from_semantic

    # Prefer code for relative phrases in the user text (LLM ISO is often off by a day).
    if user_text:
        match = _DATE_IN_QUERY_RE.search(user_text)
        if match:
            date_range = resolve_relative_date_phrase(match.group(1), today=today) or date_range

    # Move amenity-like semantics that are clearly amenities into amenities if empty
    # (keep quiet/kids as semantic)

    # Strip date-like numeric junk (field containing date)
    cleaned_numeric: list[Any] = []
    for item in numeric:
        if isinstance(item, dict):
            field = str(item.get("field") or "").lower()
            if "date" in field or field in {"start", "end", "night", "check_in"}:
                if date_range is None:
                    val = item.get("value")
                    date_range = _normalize_date_field(val, today=today) or date_range
                continue
        cleaned_numeric.append(item)

    out: dict[str, Any] = {
        "date": date_range,
        "amenities": amenities,
        "numeric_constraints": cleaned_numeric,
        "semantic_constraints": semantic,
    }
    return out


def amenity_search_queries(amenities: list[Any]) -> list[list[str]]:
    """
    For each AND group, return query strings (OR within the group).

    No ontology expansion here — place→type parents are added by the extract LLM at ingest.
    """
    groups: list[list[str]] = []
    for item in amenities or []:
        if isinstance(item, str):
            text = item.strip()
            if text:
                groups.append([text])
        elif isinstance(item, dict) and str(item.get("op", "")).lower() == "or":
            values = [
                str(v).strip() for v in (item.get("values") or []) if str(v).strip()
            ]
            if values:
                groups.append(values)
        elif isinstance(item, dict) and item.get("query"):
            text = str(item["query"]).strip()
            if text:
                groups.append([text])
    return groups


EMPTY_CONSTRAINTS: dict[str, Any] = {
    "date": None,
    "amenities": [],
    "numeric_constraints": [],
    "semantic_constraints": [],
}
