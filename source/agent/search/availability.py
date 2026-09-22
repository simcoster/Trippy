"""Open-slot vacancies and the list-price fallback quote."""

from __future__ import annotations

import os
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any, NamedTuple

from langsmith import traceable
from psycopg.rows import dict_row

from db.connect import connect
from db.experiments import table_name
from source.agent.constraints import quote_party
from source.agent.dates import iso_day, stay_night_starts
from source.agent.search.sandbox import (
    _ListPriceQuoteKey,
    _quote_caches,
    _sandbox_key_for_slot,
    _sandbox_quotes_for_slots,
    _sandbox_request_id,
    _SandboxQuoteBatch,
    _slot_rate_period,
)
from source.agent.search.sql import _attach_run_sql, _render_sql
from source.agent.timing import stage
from source.scraper.info_site.quote import quote_night
from source.scraper.info_site.schemas import RatePeriod

OPEN_SLOTS_LIMIT = 80
AVAILABILITY_TABLE_ENV = "TRIPPY_AVAILABILITY_TABLE"
_LAST_OPEN_SLOTS_QUERY: dict[str, Any] | None = None

_STAY_START = "stay_start"
_STAY_END = "stay_end"
_SITE_ID = "site_id"
_CAMPSITE = "campsite"
_ROOM_COUNT = "room_count"
_TYPE_ID = "type_id"
_TYPE_NAME = "type_name"
_MAX_OCCUPANCY = "max_occupancy"
_PARENT_ID = "parent_id"


def _availability_relation() -> str:
    """Unqualified availability table. Benchmark sets `availability_frozen`."""
    raw = (os.environ.get(AVAILABILITY_TABLE_ENV) or "availability").strip()
    return table_name(raw)


def _record_open_slots_query(record: dict[str, Any]) -> dict[str, Any]:
    global _LAST_OPEN_SLOTS_QUERY
    _LAST_OPEN_SLOTS_QUERY = record
    return record


class _StayWindow(NamedTuple):
    start: date
    end: date
    night_count: int


def _stay_window(date_range: dict) -> _StayWindow | None:
    nights = stay_night_starts(date_range)
    if not nights:
        return None
    return _StayWindow(
        start=nights[0],
        end=nights[-1] + timedelta(days=1),
        night_count=len(nights),
    )


def _open_slots_sql(
    *,
    windows: list[dict],
    site_id: int | list[int] | None,
    party_size: int | None,
    limit: int,
) -> tuple[str, list[Any]] | tuple[None, str]:
    """One query: each window must cover every night in its own range."""
    stays = [
        stay
        for window in windows
        if isinstance(window, dict) and (stay := _stay_window(window)) is not None
    ]
    if not stays:
        return None, "no_date"
    filters: list[str] = []
    filter_params: list[Any] = []
    if isinstance(site_id, list):
        ids = [int(x) for x in site_id]
        if not ids:
            return None, "empty_site_ids"
        filters.append(f"a.{_SITE_ID} = ANY(%s)")
        filter_params.append(ids)
    elif site_id is not None:
        filters.append(f"a.{_SITE_ID} = %s")
        filter_params.append(int(site_id))
    if party_size is not None:
        filters.append(
            f"(at.{_MAX_OCCUPANCY} IS NULL OR at.{_MAX_OCCUPANCY} >= %s)"
        )
        filter_params.append(int(party_size))
    where = ""
    if filters:
        where = "  WHERE " + " AND ".join(filters) + "\n"
    rel = _availability_relation()
    sql = (
        f"SELECT {_STAY_START}, {_STAY_END}, {_SITE_ID}, {_CAMPSITE}, {_ROOM_COUNT},\n"
        f"       {_TYPE_ID}, {_TYPE_NAME}, {_MAX_OCCUPANCY}, {_PARENT_ID}\n"
        "FROM (\n"
        f"  SELECT w.{_STAY_START}, w.{_STAY_END}, a.{_SITE_ID}, c.name AS {_CAMPSITE},\n"
        f"         MIN(a.{_ROOM_COUNT}) AS {_ROOM_COUNT}, at.id AS {_TYPE_ID},\n"
        f"         at.name AS {_TYPE_NAME}, at.{_MAX_OCCUPANCY}, c.{_PARENT_ID},\n"
        "         ROW_NUMBER() OVER (\n"
        f"           PARTITION BY w.{_STAY_START}, w.{_STAY_END} ORDER BY at.id\n"
        "         ) AS rn\n"
        "  FROM unnest(%s::date[], %s::date[], %s::int[])\n"
        f"    AS w({_STAY_START}, {_STAY_END}, night_count)\n"
        f"  JOIN {rel} a\n"
        f"    ON a.start_date >= w.{_STAY_START}\n"
        f"   AND a.start_date < w.{_STAY_END}\n"
        "   AND a.end_date = a.start_date + 1\n"
        "  JOIN accommodation_types at ON at.id = a.accommodation_type_id\n"
        f"  JOIN campsites c ON c.id = a.{_SITE_ID}\n"
        f"{where}"
        f"  GROUP BY w.{_STAY_START}, w.{_STAY_END}, w.night_count,\n"
        f"           a.{_SITE_ID}, c.name, at.id, at.name, at.{_MAX_OCCUPANCY},\n"
        f"           c.{_PARENT_ID}\n"
        "  HAVING COUNT(DISTINCT a.start_date) = w.night_count\n"
        ") q\n"
        "WHERE rn <= %s\n"
        f"ORDER BY {_STAY_START}, {_TYPE_ID}"
    )
    params: list[Any] = [
        [stay.start for stay in stays],
        [stay.end for stay in stays],
        [stay.night_count for stay in stays],
        *filter_params,
        limit,
    ]
    return sql, params


def _price_per_night_constraint(
    numeric: list | None,
) -> tuple[str, float] | None:
    for item in numeric or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").lower()
        if field not in {"price_per_night", "price", "cost"}:
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            return None
        op = str(item.get("operator") or "=")
        return op, value
    return None


def _price_matches(price: float | None, constraint: tuple[str, float] | None) -> bool:
    if constraint is None:
        return True
    if price is None:
        return False
    op, bound = constraint
    if op in {"<=", "=<"}:
        return price <= bound
    if op in {">=", "=>"}:
        return price >= bound
    if op == "<":
        return price < bound
    if op == ">":
        return price > bound
    return price == bound


def _load_list_prices(type_ids: list[int]) -> dict[int, list[SimpleNamespace]]:
    if not type_ids:
        return {}
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return {}
    sql = """
        SELECT at.id, lp.guest_type, lp.rate_period, lp.price
        FROM accommodation_types at
        JOIN list_prices lp ON lp.info_website_name_id = at.info_website_name_id
        WHERE at.id = ANY(%s)
    """
    try:
        with stage("sql"):
            with connect(db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (type_ids,))
                    rows = cur.fetchall()
    except Exception:
        return {}
    by_type: dict[int, list[SimpleNamespace]] = {}
    for type_id, guest_type, rate_period, price in rows:
        by_type.setdefault(int(type_id), []).append(
            SimpleNamespace(
                guest_type=guest_type,
                rate_period=rate_period,
                price=float(price),
            )
        )
    return by_type


def _quote_slot_price(
    rates: list[SimpleNamespace],
    *,
    party_size: int | None,
    rate_period: RatePeriod,
    accommodation_type_id: int | None = None,
    child_num: int | None = None,
    child_ages: tuple[int, ...] | list[int] | None = None,
) -> float | None:
    party = quote_party(
        party_size=party_size,
        child_num=child_num,
        child_ages=child_ages,
    )
    adults = party.adults_num
    weekend = rate_period == "weekend_holiday"
    cache_key: _ListPriceQuoteKey | None = None
    if accommodation_type_id is not None:
        cache_key = _ListPriceQuoteKey(
            accommodation_type_id=accommodation_type_id,
            adults_num=adults,
            weekend=weekend,
            child_num=party.child_num,
        )
        caches = _quote_caches.get()
        if caches is not None and cache_key in caches.list_price:
            return caches.list_price[cache_key]
    if not rates:
        price: float | None = None
    else:
        try:
            price = float(
                quote_night(
                    rates,
                    adults=adults,
                    children=party.child_num,
                    rate_period=rate_period,
                )
            )
        except ValueError:
            price = None
    caches = _quote_caches.get()
    if cache_key is not None and caches is not None:
        caches.list_price[cache_key] = price
    return price


@traceable(name="search_open_slots", run_type="tool")
def search_open_slots(
    *,
    date_range: dict | None = None,
    date_windows: list[dict] | None = None,
    site_id: int | list[int] | None = None,
    party_size: int | None = None,
    numeric_constraints: list | None = None,
    limit: int = OPEN_SLOTS_LIMIT,
) -> list[dict]:
    """Catalog vacancies for every stay window in one query.

    `date_range` is a one-item `date_windows`. Availability is one-night
    rows; a stay matches only when the type has a row for every night in
    [start, end). Party size uses accommodation max_occupancy (scrape is
    1-adult). Optional site_id narrows to a named park.
    `TRIPPY_AVAILABILITY_TABLE` selects the occupancy relation
    (`availability_frozen` for the planner benchmark). This does not quote.
    `numeric_constraints` is recorded on the query; `quote_open_slots`
    applies a price limit.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        _record_open_slots_query({"skipped": "no_database_url"})
        return []
    if date_windows is not None:
        windows = [item for item in date_windows if isinstance(item, dict)]
    elif isinstance(date_range, dict) and date_range.get("start"):
        windows = [date_range]
    else:
        windows = []
    if not windows:
        _record_open_slots_query({"skipped": "no_date"})
        return []
    built = _open_slots_sql(
        windows=windows,
        site_id=site_id,
        party_size=party_size,
        limit=limit,
    )
    if built[0] is None:
        _record_open_slots_query({"skipped": built[1]})
        return []
    sql, params = built
    query_record: dict[str, Any] = {
        "sql": _render_sql(sql, params),
        "price_constraint": _price_per_night_constraint(numeric_constraints),
        "windows": [
            {"start": item.get("start"), "end": item.get("end")}
            for item in windows
        ],
    }
    _record_open_slots_query(query_record)
    _attach_run_sql(sql, params)
    try:
        with stage("sql"):
            with connect(db_url) as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
    except Exception as e:
        query_record["error"] = str(e)
        return [{"error": f"Error searching availability: {e}"}]
    query_record["row_count"] = len(rows)

    slots: list[dict] = []
    for row in rows:
        occupancy_raw = row[_MAX_OCCUPANCY]
        occupancy = int(occupancy_raw) if occupancy_raw is not None else None
        parent_raw = row[_PARENT_ID]
        slots.append(
            {
                "campsite_id": int(row[_SITE_ID]),
                _CAMPSITE: row[_CAMPSITE],
                "start": iso_day(row[_STAY_START]),
                "end": iso_day(row[_STAY_END]),
                _ROOM_COUNT: int(row[_ROOM_COUNT]),
                "accommodation_type_id": int(row[_TYPE_ID]),
                "accommodation_type": row[_TYPE_NAME],
                _MAX_OCCUPANCY: occupancy,
                "occupancy_unknown": occupancy is None,
                _PARENT_ID: int(parent_raw) if parent_raw is not None else None,
            }
        )
    return slots


class _GatheredQuotes(NamedTuple):
    prices: dict[int, list[SimpleNamespace]]
    batch: _SandboxQuoteBatch


def _gather_slot_quotes(
    slots: list[dict],
    *,
    party_size: int | None,
    rate_period: RatePeriod,
    planned_entry_time: str | None,
    planned_exit_time: str | None,
    child_num: int | None,
    child_ages: tuple[int, ...] | list[int] | None,
) -> _GatheredQuotes:
    """List prices plus the sandbox batch. The planner decides when this runs."""
    type_ids = list({int(slot["accommodation_type_id"]) for slot in slots})
    prices = _load_list_prices(type_ids)
    batch = _sandbox_quotes_for_slots(
        slots,
        party_size=party_size,
        rate_period=rate_period,
        planned_entry_time=planned_entry_time,
        planned_exit_time=planned_exit_time,
        child_num=child_num,
        child_ages=child_ages,
    )
    return _GatheredQuotes(prices=prices, batch=batch)


def quote_open_slots(
    slots: list[dict],
    *,
    party_size: int | None,
    numeric_constraints: list | None = None,
    planned_entry_time: str | None = None,
    planned_exit_time: str | None = None,
    child_num: int | None = None,
    child_ages: tuple[int, ...] | list[int] | None = None,
) -> list[dict]:
    """Price vacancy rows.

    Copies them so a concurrent retrieve can keep the originals.
    """
    rate_period: RatePeriod = "weekday"
    price_constraint = _price_per_night_constraint(numeric_constraints)
    working = [dict(slot) for slot in slots]
    gathered = _gather_slot_quotes(
        working,
        party_size=party_size,
        rate_period=rate_period,
        planned_entry_time=planned_entry_time,
        planned_exit_time=planned_exit_time,
        child_num=child_num,
        child_ages=child_ages,
    )
    record = _LAST_OPEN_SLOTS_QUERY
    if isinstance(record, dict):
        record["sandbox"] = gathered.batch.report
    sandbox_quotes = gathered.batch.by_key
    party = quote_party(
        party_size=party_size,
        child_num=child_num,
        child_ages=child_ages,
    )
    adults = party.adults_num
    calls_by_id = {
        str(row["request_id"]): row
        for row in gathered.batch.report.get("calls") or []
        if isinstance(row, dict) and row.get("request_id") is not None
    }
    quoted: list[dict] = []
    for slot in working:
        slot.pop(_PARENT_ID, None)
        slot_rate = _slot_rate_period(slot, rate_period)
        key = _sandbox_key_for_slot(
            slot,
            adults=adults,
            fallback_rate=rate_period,
            planned_entry_time=planned_entry_time,
            child_num=party.child_num,
            child_ages=party.child_ages,
            planned_exit_time=planned_exit_time,
        )
        sandbox = sandbox_quotes.get(key)
        if sandbox is not None:
            price: float | None = float(sandbox.price)
            slot["price_explanation"] = sandbox.explanation
            source = "sandbox"
        else:
            price = _quote_slot_price(
                gathered.prices.get(int(slot["accommodation_type_id"])) or [],
                party_size=party_size,
                rate_period=slot_rate,
                accommodation_type_id=int(slot["accommodation_type_id"]),
                child_num=child_num,
                child_ages=child_ages,
            )
            source = "quote_night"
        traced = calls_by_id.get(_sandbox_request_id(key))
        if traced is not None:
            traced["source"] = source
            if price is not None:
                traced["price"] = price
        if not _price_matches(price, price_constraint):
            continue
        slot["price_per_night"] = price
        quoted.append(slot)
    if isinstance(record, dict):
        record["quoted_count"] = len(quoted)
    return quoted


def search_availability(
    hotel_id: int,
    *,
    date_range: dict | None = None,
    party_size: int | None = None,
    limit: int = 50,
) -> list[dict]:
    """Vacancies for one campsite (campsites.id / accommodation_types.hotel_id)."""
    return search_open_slots(
        date_range=date_range,
        site_id=hotel_id,
        party_size=party_size,
        limit=limit,
    )
