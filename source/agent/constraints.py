"""Constraint schema helpers: dates and amenity OR groups."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from source.agent.dates import (
    apply_resolved_dates,
    next_friday,
    resolve_dates,
    today_il,
)
from source.agent.dates import _as_stay_range
from source.agent.dates import _parse_iso_day


_REVIEW_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%B %Y",
    "%b %Y",
)


def parse_review_day(value: Any) -> date | None:
    """Best-effort calendar day from claims.review_date (text)."""
    parsed = _parse_iso_day(value)
    if parsed is not None:
        return parsed
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in _REVIEW_DATE_FORMATS:
        try:
            return datetime.strptime(text[:32], fmt).date()
        except ValueError:
            continue
    return None


def claim_recency(
    value: Any, *, today: date | None = None
) -> tuple[str | None, int | None]:
    """Return (YYYY-MM-DD, days_ago) for a review date; both None if unknown."""
    parsed = parse_review_day(value)
    if parsed is None:
        return None, None
    return parsed.isoformat(), (today_il(today) - parsed).days


def _normalize_semantic_item(item: Any) -> dict[str, Any] | None:
    """Normalize one semantic / leftover amenity item to {query} or {op: or, values}."""
    if item is None:
        return None
    if isinstance(item, str):
        text = item.strip()
        return {"query": text} if text else None
    if isinstance(item, dict):
        op = str(item.get("op") or "").strip().lower()
        values = item.get("values")
        if isinstance(values, list):
            cleaned = [str(v).strip() for v in values if str(v).strip()]
            if not cleaned:
                return None
            if op == "or" or (not op and len(cleaned) > 1):
                return {"op": "or", "values": cleaned}
            return {"query": cleaned[0]}
        q = str(item.get("query") or item.get("text") or "").strip()
        return {"query": q} if q else None
    text = str(item).strip()
    return {"query": text} if text else None


def _normalize_semantic_list(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        norm = _normalize_semantic_item(item)
        if norm is None:
            continue
        key = json.dumps(norm, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


def _intent_nonempty(intent: Any) -> bool:
    if not isinstance(intent, dict):
        return False
    return any(v not in (None, "", [], {}) for v in intent.values())


def _windows_from_date_field(raw: Any, *, today: date | None = None) -> list[dict[str, str]]:
    if raw is None or raw == "" or raw == []:
        return []
    if isinstance(raw, list):
        windows: list[dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict) and (item.get("start") or item.get("kind")):
                windows.extend(_windows_from_date_field(item, today=today))
            else:
                day = _parse_iso_day(item)
                if day is not None:
                    windows.append(_as_stay_range(day))
        return windows
    if isinstance(raw, dict):
        if _intent_nonempty(
            {k: raw.get(k) for k in ("kind", "weekday", "when", "weeks_from_now", "horizon_days", "on")}
        ) and not raw.get("start"):
            return list(resolve_dates(**raw, today=today).get("windows") or [])
        one = _normalize_date_field(raw, today=today)
        return [one] if one else []
    one = _normalize_date_field(raw, today=today)
    return [one] if one else []


def _normalize_date_field(raw: Any, *, today: date | None = None) -> dict[str, str] | None:
    if raw is None or raw == "" or raw == []:
        return None
    if isinstance(raw, dict):
        start = _parse_iso_day(raw.get("start") or raw.get("from"))
        end = _parse_iso_day(raw.get("end") or raw.get("to") or raw.get("start") or raw.get("from"))
        if start and end:
            if end < start:
                start, end = end, start
            return _as_stay_range(start, end)
        single = _parse_iso_day(raw.get("date") or raw.get("day"))
        if single:
            return _as_stay_range(single)
        return None
    if isinstance(raw, list):
        days = [_parse_iso_day(x) for x in raw]
        days = [d for d in days if d is not None]
        if not days:
            return None
        return _as_stay_range(min(days), max(days))
    if isinstance(raw, str):
        iso = _parse_iso_day(raw)
        if iso:
            return _as_stay_range(iso)
        return None
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
    resolved: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Normalize extractor output to:
      date: {start, end} | null  (first window)
      date_windows: [{start, end}, ...]
      date_truncated / date_notice
      numeric_constraints: [...]
      semantic_constraints: [{query: ...} | {op: or, values: [...]}, ...]

    Legacy `amenities` keys are folded into semantic_constraints.
    """
    parsed = parse_constraints_dict(data)
    today = today_il(today)
    _ = user_text

    semantic = _normalize_semantic_list(
        list(parsed.get("semantic_constraints") or [])
        + list(parsed.get("amenities") or [])
    )
    numeric = list(parsed.get("numeric_constraints") or [])
    semantic = _normalize_semantic_list(semantic)

    intent = parsed.get("date_intent")
    used_resolved = resolved if isinstance(resolved, dict) else None
    if used_resolved is None and _intent_nonempty(intent):
        used_resolved = resolve_dates(**intent, today=today)

    if used_resolved is None or not used_resolved.get("windows"):
        existing = parsed.get("date_windows")
        if isinstance(existing, list) and existing:
            used_resolved = {
                "windows": _windows_from_date_field(existing, today=today),
                "truncated": bool(parsed.get("date_truncated")),
                "notice": parsed.get("date_notice"),
            }
        else:
            iso_windows = _windows_from_date_field(parsed.get("date"), today=today)
            if iso_windows:
                used_resolved = {
                    "windows": iso_windows,
                    "truncated": False,
                    "notice": None,
                }

    # Strip date-like numeric junk (field containing date)
    cleaned_numeric: list[Any] = []
    for item in numeric:
        if isinstance(item, dict):
            field = str(item.get("field") or "").lower()
            if "date" in field or field in {"start", "end", "night", "check_in"}:
                if not (used_resolved and used_resolved.get("windows")):
                    val = item.get("value")
                    one = _normalize_date_field(val, today=today)
                    if one:
                        used_resolved = {
                            "windows": [one],
                            "truncated": False,
                            "notice": None,
                        }
                continue
        cleaned_numeric.append(item)

    out = {
        "numeric_constraints": cleaned_numeric,
        "semantic_constraints": semantic,
    }
    apply_resolved_dates(out, used_resolved)
    return out


def campsite_name_from_parsed(parsed: dict[str, Any] | None) -> str | None:
    """Named park the user asked to stay at — not a region vibe, not the catalog."""
    if not parsed:
        return None
    raw = parsed.get("campsite")
    if isinstance(raw, dict):
        raw = raw.get("name") or raw.get("query") or raw.get("text")
    text = str(raw or "").strip()
    return text or None


def semantic_search_queries(constraints: list[Any]) -> list[list[str]]:
    """
    For each AND group, return query strings (OR within the group).

    No ontology expansion here — place→type parents are added by the extract LLM at ingest.
    """
    groups: list[list[str]] = []
    for item in constraints or []:
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


# Back-compat alias: amenities were folded into semantic_constraints.
amenity_search_queries = semantic_search_queries


EMPTY_CONSTRAINTS: dict[str, Any] = {
    "date": None,
    "date_windows": [],
    "date_truncated": False,
    "date_notice": None,
    "numeric_constraints": [],
    "semantic_constraints": [],
}
