"""One price-sandbox quote for the slots of a user request."""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, NamedTuple

from langsmith import traceable

from source.agent.constraints import quote_party
from source.agent.dates import _parse_iso_day
from source.price_sandbox.client import (
    QuoteRequest,
    quote_replies,
    sandbox_reachable,
    sandbox_url,
)
from source.price_sandbox.params import QuoteParams, QuoteResult
from source.scraper.info_site.schemas import RatePeriod


class _SandboxQuoteKey(NamedTuple):
    campsite_id: int
    lodging: str
    adults_num: int
    weekend: bool
    planned_entry_time: str | None = None
    child_num: int = 0
    child_ages: tuple[int, ...] = ()
    planned_exit_time: str | None = None


class _SandboxQuoteBatch(NamedTuple):
    by_key: dict[_SandboxQuoteKey, QuoteResult]
    report: dict[str, Any]


class _ListPriceQuoteKey(NamedTuple):
    accommodation_type_id: int
    adults_num: int
    weekend: bool
    child_num: int = 0


class _QuoteCaches(NamedTuple):
    sandbox: dict[_SandboxQuoteKey, QuoteResult]
    list_price: dict[_ListPriceQuoteKey, float | None]


_quote_caches: contextvars.ContextVar[_QuoteCaches | None] = contextvars.ContextVar(
    "trippy_price_quote_caches", default=None
)


@contextmanager
def price_quote_cache() -> Any:
    """One jail/list-price memo for this user request. Next request starts empty."""
    token = _quote_caches.set(_QuoteCaches(sandbox={}, list_price={}))
    try:
        yield
    finally:
        _quote_caches.reset(token)


def clear_price_quote_cache() -> None:
    """Empty the current request cache, if a request is open."""
    caches = _quote_caches.get()
    if caches is None:
        return
    caches.sandbox.clear()
    caches.list_price.clear()


def _rate_period_for_stay(date_range: dict | None) -> RatePeriod:
    if not isinstance(date_range, dict):
        return "weekday"
    start = _parse_iso_day(date_range.get("start"))
    if start is None:
        return "weekday"
    end = _parse_iso_day(date_range.get("end")) or (start + timedelta(days=1))
    day = start
    while day < end:
        if day.weekday() >= 5:
            return "weekend_holiday"
        day += timedelta(days=1)
    return "weekday"


@traceable(name="price_sandbox_quote", run_type="tool")
def _quote_sandbox_batch(
    *,
    url: str | None,
    calls: list[dict[str, Any]],
    skip: str | None = None,
) -> list[dict[str, Any]]:
    """POST /quote. LangSmith inputs/outputs are the per-campsite rows."""
    if skip:
        return [{"skipped": skip}] if not calls else [
            {**call, "ok": False, "error": skip} for call in calls
        ]
    requests = [
        QuoteRequest(
            request_id=str(call["request_id"]),
            site_id=int(call["campsite_id"]),
            parent_site_id=(
                int(call["parent_site_id"])
                if call.get("parent_site_id") is not None
                else None
            ),
            params=QuoteParams(
                lodging=str(call["lodging"]),
                adults_num=int(call["adults_num"]),
                child_num=int(call.get("child_num") or 0),
                child_ages=tuple(int(age) for age in (call.get("child_ages") or ())),
                is_weekend_or_holiday=bool(call["is_weekend_or_holiday"]),
                planned_entry_time=(
                    str(call["planned_entry_time"]).strip()
                    if call.get("planned_entry_time")
                    else None
                ),
                planned_exit_time=(
                    str(call["planned_exit_time"]).strip()
                    if call.get("planned_exit_time")
                    else None
                ),
            ),
        )
        for call in calls
    ]
    try:
        replies = quote_replies(requests, base_url=url)
    except Exception as exc:
        return [{**call, "ok": False, "error": str(exc)} for call in calls]
    out: list[dict[str, Any]] = []
    for call, reply in zip(calls, replies):
        if reply.ok:
            out.append(
                {
                    **call,
                    "ok": True,
                    "price": reply.price,
                    "explanation": reply.explanation,
                }
            )
            continue
        out.append({**call, "ok": False, "error": reply.error or "quote_failed"})
    return out


def _sandbox_request_id(key: _SandboxQuoteKey) -> str:
    req_id = (
        f"{key.campsite_id}:{key.lodging}:{key.adults_num}:{int(key.weekend)}"
    )
    if key.planned_entry_time:
        req_id = f"{req_id}:{key.planned_entry_time}"
    if key.planned_exit_time:
        req_id = f"{req_id}:x{key.planned_exit_time}"
    if key.child_num or key.child_ages:
        ages = ",".join(str(age) for age in key.child_ages)
        req_id = f"{req_id}:c{key.child_num}:{ages}"
    return req_id


def _slot_rate_period(slot: dict, fallback: RatePeriod) -> RatePeriod:
    """Weekend follows the slot's own nights when it has dates."""
    if slot.get("start"):
        return _rate_period_for_stay(
            {"start": slot["start"], "end": slot.get("end")}
        )
    return fallback


def _sandbox_key_for_slot(
    slot: dict,
    *,
    adults: int,
    fallback_rate: RatePeriod,
    planned_entry_time: str | None,
    child_num: int = 0,
    child_ages: tuple[int, ...] = (),
    planned_exit_time: str | None = None,
) -> _SandboxQuoteKey:
    entry = str(planned_entry_time).strip() if planned_entry_time else None
    exit_at = str(planned_exit_time).strip() if planned_exit_time else None
    return _SandboxQuoteKey(
        campsite_id=int(slot["campsite_id"]),
        lodging=str(slot.get("accommodation_type") or ""),
        adults_num=adults,
        weekend=_slot_rate_period(slot, fallback_rate) == "weekend_holiday",
        planned_entry_time=entry,
        child_num=child_num,
        child_ages=child_ages,
        planned_exit_time=exit_at,
    )


def _sandbox_quotes_for_slots(
    slots: list[dict],
    *,
    party_size: int | None,
    rate_period: RatePeriod,
    planned_entry_time: str | None = None,
    planned_exit_time: str | None = None,
    child_num: int | None = None,
    child_ages: tuple[int, ...] | list[int] | None = None,
) -> _SandboxQuoteBatch:
    from source.agent.search.availability import _CAMPSITE, _PARENT_ID

    url = sandbox_url()
    party = quote_party(
        party_size=party_size,
        child_num=child_num,
        child_ages=child_ages,
    )
    adults = party.adults_num
    calls: list[dict[str, Any]] = []
    keys: list[_SandboxQuoteKey] = []
    seen_keys: set[_SandboxQuoteKey] = set()
    for slot in slots:
        key = _sandbox_key_for_slot(
            slot,
            adults=adults,
            fallback_rate=rate_period,
            planned_entry_time=planned_entry_time,
            child_num=party.child_num,
            child_ages=party.child_ages,
            planned_exit_time=planned_exit_time,
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        keys.append(key)
        calls.append(
            {
                "request_id": _sandbox_request_id(key),
                "campsite_id": key.campsite_id,
                "campsite": slot.get(_CAMPSITE),
                "lodging": key.lodging,
                "adults_num": key.adults_num,
                "child_num": key.child_num,
                "child_ages": list(key.child_ages),
                "is_weekend_or_holiday": key.weekend,
                "planned_entry_time": key.planned_entry_time,
                "planned_exit_time": key.planned_exit_time,
                "parent_site_id": slot.get(_PARENT_ID),
            }
        )
    by_key: dict[_SandboxQuoteKey, QuoteResult] = {}
    fresh_keys: list[_SandboxQuoteKey] = []
    fresh_calls: list[dict[str, Any]] = []
    cached_rows: list[dict[str, Any]] = []
    sandbox_cache = None
    caches = _quote_caches.get()
    if caches is not None:
        sandbox_cache = caches.sandbox
    for key, call in zip(keys, calls):
        hit = sandbox_cache.get(key) if sandbox_cache is not None else None
        if hit is not None:
            by_key[key] = hit
            cached_rows.append(
                {
                    **call,
                    "ok": True,
                    "price": hit.price,
                    "explanation": hit.explanation,
                    "cached": True,
                }
            )
            continue
        fresh_keys.append(key)
        fresh_calls.append(call)
    skip: str | None = None
    if not slots:
        skip = "no_slots"
    elif fresh_calls and not url:
        skip = "PRICE_SANDBOX_URL unset"
    elif fresh_calls and not sandbox_reachable(base_url=url):
        skip = "sandbox not reachable"
    started = time.perf_counter()
    if skip:
        rows = _quote_sandbox_batch(url=url, calls=fresh_calls, skip=skip)
    elif fresh_calls:
        rows = _quote_sandbox_batch(url=url, calls=fresh_calls)
        for key, row in zip(fresh_keys, rows):
            if row.get("ok"):
                result = QuoteResult(
                    price=float(row["price"]),
                    explanation=str(row.get("explanation") or ""),
                )
                if sandbox_cache is not None:
                    sandbox_cache[key] = result
                by_key[key] = result
    else:
        rows = []
    report = {
        "url": url,
        "skipped": skip,
        "calls": cached_rows + rows,
        "cached": len(cached_rows),
        "latency_ms": (time.perf_counter() - started) * 1000,
    }
    return _SandboxQuoteBatch(by_key=by_key, report=report)
