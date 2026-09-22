"""Catalog, availability, amenity, and claims search — not graph wiring."""

from __future__ import annotations

import contextvars
import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any, NamedTuple

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langsmith import traceable
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field

from db.connect import connect
from db.experiments import table_name
from db.models import SubjectCategory
from source.agent.constraints import claim_recency, today_il
from source.agent.dates import _parse_iso_day, iso_day, stay_night_starts
from source.agent.timing import stage
from source.agent.tracing import tracing_env_on
from source.price_sandbox.client import (
    QuoteRequest,
    quote_replies,
    sandbox_reachable,
    sandbox_url,
)
from source.price_sandbox.params import QuoteParams, QuoteResult
from source.scraper.amenity_enrichment.llm import ClaimsEmbeddingLLMClient
from source.scraper.info_site.quote import quote_night
from source.scraper.info_site.schemas import RatePeriod

load_dotenv()


class _SandboxQuoteKey(NamedTuple):
    campsite_id: int
    lodging: str
    adults_num: int
    weekend: bool
    planned_entry_time: str | None = None


class _SandboxQuoteBatch(NamedTuple):
    by_key: dict[_SandboxQuoteKey, QuoteResult]
    report: dict[str, Any]


class _ListPriceQuoteKey(NamedTuple):
    accommodation_type_id: int
    adults_num: int
    weekend: bool


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

_claims_embedder = ClaimsEmbeddingLLMClient()
QUERY_EMBED_CONCURRENCY = 5

# Site-wide rules for a candidate: this campsite and its parent. Sister
# subcamps (Akhziv north vs south) do not share each other's rows.
_OWN_OR_PARENT_RULES = (
    "(cr.campsite_id = {alias}.id OR cr.campsite_id = {alias}.parent_id)"
)

OPEN_SLOTS_LIMIT = 80
AVAILABILITY_TABLE_ENV = "TRIPPY_AVAILABILITY_TABLE"
_LAST_OPEN_SLOTS_QUERY: dict[str, Any] | None = None
# pg_trgm: typos on a similar-length name, and a short query inside a long
# Hebrew title. 0.4 is below the default word_similarity threshold (0.6) so
# Achziv/Akhziv still ranks; LIMIT 5 is the cap.
NAME_LOOKUP_MIN_SCORE = 0.4
NAME_LOOKUP_LIMIT = 5

LOOKUP_CAMPSITE_SQL = """
SELECT id, name, english_name, booking_hotel_id, score
FROM (
    SELECT id, name, english_name, booking_hotel_id,
           GREATEST(
               similarity(%(q)s, name),
               similarity(%(q)s, COALESCE(english_name, '')),
               word_similarity(%(q)s, name),
               word_similarity(%(q)s, COALESCE(english_name, ''))
           ) AS score
    FROM campsites
) ranked
WHERE score >= %(min_score)s
ORDER BY score DESC, id
LIMIT %(limit)s
"""


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
        filters.append("a.site_id = ANY(%s)")
        filter_params.append(ids)
    elif site_id is not None:
        filters.append("a.site_id = %s")
        filter_params.append(int(site_id))
    if party_size is not None:
        filters.append("(at.max_occupancy IS NULL OR at.max_occupancy >= %s)")
        filter_params.append(int(party_size))
    where = ""
    if filters:
        where = "  WHERE " + " AND ".join(filters) + "\n"
    rel = _availability_relation()
    sql = (
        "SELECT stay_start, stay_end, site_id, campsite, room_count,\n"
        "       type_id, type_name, max_occupancy, parent_id\n"
        "FROM (\n"
        "  SELECT w.stay_start, w.stay_end, a.site_id, c.name AS campsite,\n"
        "         MIN(a.room_count) AS room_count, at.id AS type_id,\n"
        "         at.name AS type_name, at.max_occupancy, c.parent_id,\n"
        "         ROW_NUMBER() OVER (\n"
        "           PARTITION BY w.stay_start, w.stay_end ORDER BY at.id\n"
        "         ) AS rn\n"
        "  FROM unnest(%s::date[], %s::date[], %s::int[])\n"
        "    AS w(stay_start, stay_end, night_count)\n"
        f"  JOIN {rel} a\n"
        "    ON a.start_date >= w.stay_start\n"
        "   AND a.start_date < w.stay_end\n"
        "   AND a.end_date = a.start_date + 1\n"
        "  JOIN accommodation_types at ON at.id = a.accommodation_type_id\n"
        "  JOIN campsites c ON c.id = a.site_id\n"
        f"{where}"
        "  GROUP BY w.stay_start, w.stay_end, w.night_count,\n"
        "           a.site_id, c.name, at.id, at.name, at.max_occupancy,\n"
        "           c.parent_id\n"
        "  HAVING COUNT(DISTINCT a.start_date) = w.night_count\n"
        ") q\n"
        "WHERE rn <= %s\n"
        "ORDER BY stay_start, type_id"
    )
    params: list[Any] = [
        [stay.start for stay in stays],
        [stay.end for stay in stays],
        [stay.night_count for stay in stays],
        *filter_params,
        limit,
    ]
    return sql, params


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "ARRAY[" + ", ".join(_sql_literal(v) for v in value) + "]"
    if hasattr(value, "isoformat"):
        value = value.isoformat()
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _render_sql(sql: str, params: list[Any]) -> str:
    parts = sql.split("%s")
    if len(parts) != len(params) + 1:
        return sql
    out = [parts[0]]
    for part, param in zip(parts[1:], params):
        out.append(_sql_literal(param))
        out.append(part)
    return "".join(out)


def _trace_sql_param(value: Any) -> Any:
    """Keep LangSmith SQL readable; pgvector literals are thousands of floats."""
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        return "<vector>"
    return value


def _drop_embedding_input(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in inputs.items() if key != "embedding"}


def _attach_run_sql(sql: str, params: list[Any]) -> None:
    """Put interpolated SQL on the current LangSmith span, if any."""
    try:
        from langsmith import get_current_run_tree
    except ImportError:
        return
    run = get_current_run_tree()
    if run is None:
        return
    rendered = _render_sql(sql, [_trace_sql_param(p) for p in params])
    try:
        inputs = dict(run.inputs or {})
        inputs["sql"] = rendered
        run.inputs = inputs
    except Exception:
        return


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
) -> float | None:
    adults = party_size if party_size and party_size > 0 else 1
    weekend = rate_period == "weekend_holiday"
    cache_key: _ListPriceQuoteKey | None = None
    if accommodation_type_id is not None:
        cache_key = _ListPriceQuoteKey(
            accommodation_type_id=accommodation_type_id,
            adults_num=adults,
            weekend=weekend,
        )
        caches = _quote_caches.get()
        if caches is not None and cache_key in caches.list_price:
            return caches.list_price[cache_key]
    if not rates:
        price: float | None = None
    else:
        try:
            price = float(quote_night(rates, adults=adults, rate_period=rate_period))
        except ValueError:
            price = None
    caches = _quote_caches.get()
    if cache_key is not None and caches is not None:
        caches.list_price[cache_key] = price
    return price


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
                is_weekend_or_holiday=bool(call["is_weekend_or_holiday"]),
                planned_entry_time=(
                    str(call["planned_entry_time"]).strip()
                    if call.get("planned_entry_time")
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
        return f"{req_id}:{key.planned_entry_time}"
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
) -> _SandboxQuoteKey:
    entry = str(planned_entry_time).strip() if planned_entry_time else None
    return _SandboxQuoteKey(
        campsite_id=int(slot["campsite_id"]),
        lodging=str(slot.get("accommodation_type") or ""),
        adults_num=adults,
        weekend=_slot_rate_period(slot, fallback_rate) == "weekend_holiday",
        planned_entry_time=entry,
    )


def _sandbox_quotes_for_slots(
    slots: list[dict],
    *,
    party_size: int | None,
    rate_period: RatePeriod,
    planned_entry_time: str | None = None,
) -> _SandboxQuoteBatch:
    url = sandbox_url()
    adults = party_size if party_size and party_size > 0 else 1
    calls: list[dict[str, Any]] = []
    keys: list[_SandboxQuoteKey] = []
    seen_keys: set[_SandboxQuoteKey] = set()
    for slot in slots:
        key = _sandbox_key_for_slot(
            slot,
            adults=adults,
            fallback_rate=rate_period,
            planned_entry_time=planned_entry_time,
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        keys.append(key)
        calls.append(
            {
                "request_id": _sandbox_request_id(key),
                "campsite_id": key.campsite_id,
                "campsite": slot.get("campsite"),
                "lodging": key.lodging,
                "adults_num": key.adults_num,
                "is_weekend_or_holiday": key.weekend,
                "planned_entry_time": key.planned_entry_time,
                "parent_site_id": slot.get("parent_id"),
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


@traceable(name="search_open_slots", run_type="tool")
def search_open_slots(
    *,
    date_range: dict | None = None,
    date_windows: list[dict] | None = None,
    site_id: int | list[int] | None = None,
    party_size: int | None = None,
    numeric_constraints: list | None = None,
    planned_entry_time: str | None = None,
    limit: int = OPEN_SLOTS_LIMIT,
) -> list[dict]:
    """Catalog vacancies for every stay window in one query.

    `date_range` is a one-item `date_windows`. Availability is one-night
    rows; a stay matches only when the type has a row for every night in
    [start, end). One sandbox quote covers the rows. Party size uses
    accommodation max_occupancy (scrape is 1-adult). Price filters use
    quote_night against list_prices. Optional site_id narrows to a named
    park. `TRIPPY_AVAILABILITY_TABLE` selects the occupancy relation
    (`availability_frozen` for the planner benchmark).
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
    price_constraint = _price_per_night_constraint(numeric_constraints)
    rate_period: RatePeriod = "weekday"
    query_record: dict[str, Any] = {
        "sql": _render_sql(sql, params),
        "price_constraint": price_constraint,
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
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
    except Exception as e:
        query_record["error"] = str(e)
        return [{"error": f"Error searching availability: {e}"}]
    query_record["row_count"] = len(rows)

    slots: list[dict] = []
    for row in rows:
        occupancy = int(row[7]) if row[7] is not None else None
        parent_raw = row[8] if len(row) > 8 else None
        slots.append(
            {
                "campsite_id": int(row[2]),
                "campsite": row[3],
                "start": iso_day(row[0]),
                "end": iso_day(row[1]),
                "room_count": int(row[4]),
                "accommodation_type_id": int(row[5]),
                "accommodation_type": row[6],
                "max_occupancy": occupancy,
                "occupancy_unknown": occupancy is None,
                "parent_id": int(parent_raw) if parent_raw is not None else None,
            }
        )
    type_ids = list({int(s["accommodation_type_id"]) for s in slots})
    prices = _load_list_prices(type_ids)
    sandbox_batch = _sandbox_quotes_for_slots(
        slots,
        party_size=party_size,
        rate_period=rate_period,
        planned_entry_time=planned_entry_time,
    )
    query_record["sandbox"] = sandbox_batch.report
    sandbox_quotes = sandbox_batch.by_key
    adults = party_size if party_size and party_size > 0 else 1
    calls_by_id = {
        str(row["request_id"]): row
        for row in sandbox_batch.report.get("calls") or []
        if isinstance(row, dict) and row.get("request_id") is not None
    }
    quoted: list[dict] = []
    for slot in slots:
        slot.pop("parent_id", None)
        slot_rate = _slot_rate_period(slot, rate_period)
        key = _sandbox_key_for_slot(
            slot,
            adults=adults,
            fallback_rate=rate_period,
            planned_entry_time=planned_entry_time,
        )
        sandbox = sandbox_quotes.get(key)
        if sandbox is not None:
            price: float | None = float(sandbox.price)
            slot["price_explanation"] = sandbox.explanation
            source = "sandbox"
        else:
            price = _quote_slot_price(
                prices.get(int(slot["accommodation_type_id"])) or [],
                party_size=party_size,
                rate_period=slot_rate,
                accommodation_type_id=int(slot["accommodation_type_id"]),
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
    query_record["quoted_count"] = len(quoted)
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


def lookup_campsite_by_name(name: str) -> list[dict]:
    """Resolve a user-named park to campsite id(s). Not a catalog dump.

    Ranks the query against Hebrew `name` and stored `english_name` with
    pg_trgm (`similarity` + `word_similarity`). Discovery fills English;
    there is no alias list.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    query = (name or "").strip()
    if not query:
        return []
    try:
        with stage("sql"):
            with connect(db_url) as conn:
                with conn.cursor() as cur:
                    return match_campsites_by_name(cur, query)
    except Exception as e:
        return [{"error": f"Error looking up campsite: {e}"}]


def match_campsites_by_name(cur: Any, query: str) -> list[dict]:
    """pg_trgm rank of `query` vs `campsites.name` / `english_name`."""
    cur.execute(
        LOOKUP_CAMPSITE_SQL,
        {
            "q": query,
            "min_score": NAME_LOOKUP_MIN_SCORE,
            "limit": NAME_LOOKUP_LIMIT,
        },
    )
    return [
        {
            "id": int(row[0]),
            "name": row[1],
            "hotel_id": int(row[0]),
            "booking_hotel_id": row[3],
        }
        for row in cur.fetchall()
    ]


def search_campsites(numeric_constraints):
    """
    List campsites from the 'campsites' table (id, name, url).
    Numeric filters (price / ride time) are not on this table yet;
    they will come from availability data later. `numeric_constraints`
    is accepted for API compatibility with the planner node.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return "Error: DATABASE_URL not configured"
    _ = numeric_constraints  # reserved for future availability filters
    sql = """
        SELECT id, name, url
        FROM campsites
        ORDER BY id
        LIMIT 50
    """
    try:
        with connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                if not rows:
                    return "No campsites found"
                return [
                    {"id": row[0], "name": row[1], "url": row[2]}
                    for row in rows
                ]
    except Exception as e:
        return f"Error during search_campsites: {e}"


def _query_vec_literal(query: str) -> str:
    with stage("embed"):
        embedding = _claims_embedder.embed([query])[0]
        return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


class _EmbedQueryArgs(BaseModel):
    query: str = Field(description="Amenity or place phrase to embed.")


def _run_embed_query_tool(query: str) -> str:
    return _query_vec_literal(query)


embed_query_tool = StructuredTool.from_function(
    func=_run_embed_query_tool,
    name="embed_query",
    description="Embed one planner retrieve query for pgvector search.",
    args_schema=_EmbedQueryArgs,
)


def _use_embed_tool() -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return tracing_env_on()


def _invoke_embed_query_tool(query: str) -> str:
    return embed_query_tool.invoke({"query": query})


@traceable(name="embed_queries", run_type="tool")
def _query_vec_literals(queries: Iterable[str]) -> dict[str, str]:
    """Embed distinct query statements, up to QUERY_EMBED_CONCURRENCY at a time."""
    unique = list(dict.fromkeys(queries))
    if not unique:
        return {}
    worker = (
        _invoke_embed_query_tool if _use_embed_tool() else _query_vec_literal
    )
    workers = min(QUERY_EMBED_CONCURRENCY, len(unique))
    if workers == 1:
        query = unique[0]
        return {query: worker(query)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if worker is _invoke_embed_query_tool:
            futs = [
                pool.submit(contextvars.copy_context().run, worker, query)
                for query in unique
            ]
            literals = [fut.result() for fut in futs]
        else:
            literals = list(pool.map(worker, unique))
    return dict(zip(unique, literals, strict=True))


@traceable(
    name="search_stated_amenities",
    run_type="tool",
    process_inputs=_drop_embedding_input,
)
def search_stated_amenities(
    query: str,
    limit: int = 5,
    *,
    embedding: str | None = None,
    accommodation_type_ids: list[int] | None = None,
) -> list[dict]:
    """Rank accommodation types by closest official amenity embedding."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    if accommodation_type_ids is not None and not accommodation_type_ids:
        return []
    vec_literal = embedding or _query_vec_literal(query)
    # Per-unit amenities are `campsite_rules` rows scoped to the type. The
    # `accommodation_types.amenities` JSONB this used to read was dropped in
    # migration 027; see docs/design.md.
    clauses = [
        "a.embedding IS NOT NULL",
        "a.category = %s",
        # An amenity array only ever held things the unit has. A NULL polarity
        # is a bare quantity, which still describes something present; only an
        # explicit false is a negative, and those were the separate
        # `not_included_amenities` array that this lane never read.
        "cr.polarity IS DISTINCT FROM false",
    ]
    params: list[Any] = [vec_literal, vec_literal, int(SubjectCategory.AMENITY)]
    if accommodation_type_ids is not None:
        clauses.append("at.id = ANY(%s)")
        params.append([int(x) for x in accommodation_type_ids])
    params.append(limit)
    sql = f"""
        SELECT at.id,
               at.name,
               at.hotel_id,
               MIN(a.embedding <#> %s::vector) AS distance,
               (array_agg(a.name ORDER BY a.embedding <#> %s::vector))[1]
                   AS matched_amenity
        FROM accommodation_types at
        JOIN campsite_rules cr ON cr.accommodation_type_id = at.id
        JOIN subject_vectors a ON a.id = cr.subject_id
        WHERE {' AND '.join(clauses)}
        GROUP BY at.id, at.name, at.hotel_id
        ORDER BY distance
        LIMIT %s
    """
    _attach_run_sql(sql, params)
    try:
        with stage("retrieve"):
            with connect(db_url) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        return [
            {
                "amenity": row[4],
                "accommodation_type_id": int(row[0]),
                "accommodation_type": row[1],
                "hotel_id": int(row[2]),
                "distance": float(row[3]),
            }
            for row in rows
        ]
    except Exception as e:
        return [{"error": f"Error searching stated amenities: {e}"}]


@traceable(
    name="search_site_amenities",
    run_type="tool",
    process_inputs=_drop_embedding_input,
)
def search_site_amenities(
    query: str,
    limit: int = 5,
    *,
    embedding: str | None = None,
    campsite_ids: list[int] | None = None,
) -> list[dict]:
    """Rank campsites by closest site-wide (communal) amenity embedding.

    A subcamp's scan includes the parent's site-wide rows, not a sister's.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    if campsite_ids is not None and not campsite_ids:
        return []
    vec_literal = embedding or _query_vec_literal(query)
    # Site-wide amenities are `campsite_rules` rows with no accommodation type.
    # `campsites.amenities` was a mirror of exactly these rows and was dropped
    # in migration 027, along with the sync step that maintained it.
    clauses = [
        "a.embedding IS NOT NULL",
        "a.category = %s",
        "cr.polarity IS DISTINCT FROM false",
    ]
    params: list[Any] = [vec_literal, vec_literal, int(SubjectCategory.AMENITY)]
    if campsite_ids is not None:
        clauses.append("c.id = ANY(%s)")
        params.append([int(x) for x in campsite_ids])
    params.append(limit)
    sql = f"""
        SELECT c.id,
               c.name,
               MIN(a.embedding <#> %s::vector) AS distance,
               (array_agg(a.name ORDER BY a.embedding <#> %s::vector))[1]
                   AS matched_amenity
        FROM campsites c
        JOIN campsite_rules cr
          ON cr.accommodation_type_id IS NULL
         AND {_OWN_OR_PARENT_RULES.format(alias="c")}
        JOIN subject_vectors a ON a.id = cr.subject_id
        WHERE {' AND '.join(clauses)}
        GROUP BY c.id, c.name
        ORDER BY distance
        LIMIT %s
    """
    _attach_run_sql(sql, params)
    try:
        with stage("retrieve"):
            with connect(db_url) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        return [
            {
                "amenity": row[3],
                "campsite_id": int(row[0]),
                "campsite": row[1],
                "distance": float(row[2]),
            }
            for row in rows
        ]
    except Exception as e:
        return [{"error": f"Error searching site amenities: {e}"}]


@traceable(
    name="search_campsite_rules",
    run_type="tool",
    process_inputs=_drop_embedding_input,
)
def search_campsite_rules(
    query: str,
    limit: int = 5,
    *,
    embedding: str | None = None,
    campsite_ids: list[int] | None = None,
) -> list[dict]:
    """Nearest official rules per campsite, all subject categories.

    Unlike the amenity lanes this includes polarity-false rows (dogs_allowed
    forbidden) and boolean/numeric rules. A subcamp reads its own rules and
    its parent's, never a sister's: ingest writes visitor-info onto the child,
    reviews live on the parent.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    if campsite_ids is not None and not campsite_ids:
        return []
    vec_literal = embedding or _query_vec_literal(query)
    ids = [int(x) for x in campsite_ids] if campsite_ids is not None else None
    if ids is None:
        sql = """
            SELECT cr.campsite_id, sv.name, sv.category, cr.polarity,
                   cr.qualifier, cr.qualifier_unit, cr.evidence_span,
                   cr.accommodation_type_id,
                   sv.embedding <#> %s::vector AS distance
            FROM campsite_rules cr
            JOIN subject_vectors sv ON sv.id = cr.subject_id
            WHERE sv.embedding IS NOT NULL
            ORDER BY sv.embedding <#> %s::vector
            LIMIT %s
        """
        params: list[Any] = [vec_literal, vec_literal, limit]
    else:
        sql = f"""
            SELECT s.campsite_id, x.name, x.category, x.polarity,
                   x.qualifier, x.qualifier_unit, x.evidence_span,
                   x.accommodation_type_id, x.distance
            FROM unnest(%s::bigint[]) AS s(campsite_id)
            JOIN campsites site ON site.id = s.campsite_id
            CROSS JOIN LATERAL (
                SELECT sv.name, sv.category, cr.polarity,
                       cr.qualifier, cr.qualifier_unit, cr.evidence_span,
                       cr.accommodation_type_id,
                       sv.embedding <#> %s::vector AS distance
                FROM campsite_rules cr
                JOIN subject_vectors sv ON sv.id = cr.subject_id
                WHERE sv.embedding IS NOT NULL
                  AND {_OWN_OR_PARENT_RULES.format(alias="site")}
                ORDER BY sv.embedding <#> %s::vector
                LIMIT %s
            ) x
        """
        params = [ids, vec_literal, vec_literal, limit]
    _attach_run_sql(sql, params)
    try:
        with stage("rules"):
            with connect(db_url) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        return [
            {
                "campsite_id": int(row[0]),
                "subject": row[1],
                "category": int(row[2]),
                "polarity": row[3],
                "qualifier": float(row[4]) if row[4] is not None else None,
                "qualifier_unit": int(row[5]) if row[5] is not None else None,
                "evidence_span": row[6],
                "accommodation_type_id": int(row[7]) if row[7] is not None else None,
                "distance": float(row[8]),
            }
            for row in rows
        ]
    except Exception as e:
        return [{"error": f"Error searching campsite rules: {e}"}]


# Global top-K over every claim. Used when no campsite scope is given.
_CLAIMS_GLOBAL_SQL = """
        SELECT c.campsite_id, c.claim, c.is_positive, r.published_at,
               c.embedding <#> %s::vector AS distance
        FROM claims c
        LEFT JOIN reviews r ON r.id = c.review_id
        WHERE c.claim IS NOT NULL
          AND (r.id IS NULL OR r.skip_reason IS NULL)
        ORDER BY c.embedding <#> %s::vector
        LIMIT %s
"""

# Top-`limit` per campsite. A global top-K crowds out a candidate site's best
# claim with another site's, so drive one scan per site instead of trimming
# after. LATERAL (not a window over the whole table) so each sub-select rides
# claim_campsite_idx.
#
# A subcamp reads its parent's claims. Reviews are written against the Google
# place, which the parent owns, and a guest says "Akhziv" rather than naming a
# subcamp — so claims only ever exist on the parent row (review-split and
# breadcrumb regions). Matching campsite_id exactly would make every subcamp
# look like a site nobody has reviewed, while the parent looked like a site
# with no amenities. The hit is still reported under the id the caller asked
# about, so a subcamp's claims rank against that subcamp's rules.
# Breadcrumb claims have review_id NULL; LEFT JOIN so they still retrieve.
_CLAIMS_BY_SITE_SQL = """
        SELECT s.campsite_id, x.claim, x.is_positive, x.published_at, x.distance
        FROM unnest(%s::bigint[]) AS s(campsite_id)
        JOIN campsites site ON site.id = s.campsite_id
        CROSS JOIN LATERAL (
            SELECT c.claim, c.is_positive, r.published_at,
                   c.embedding <#> %s::vector AS distance
            FROM claims c
            LEFT JOIN reviews r ON r.id = c.review_id
            WHERE c.campsite_id = COALESCE(site.parent_id, site.id)
              AND c.claim IS NOT NULL
              AND c.embedding IS NOT NULL
              AND (r.id IS NULL OR r.skip_reason IS NULL)
            ORDER BY c.embedding <#> %s::vector
            LIMIT %s
        ) AS x
        ORDER BY x.distance
"""


@traceable(
    name="search_review_claims",
    run_type="tool",
    process_inputs=_drop_embedding_input,
)
def search_review_claims(
    query: str,
    limit: int = 5,
    *,
    embedding: str | None = None,
    campsite_ids: list[int] | None = None,
) -> list[dict]:
    """Search review claims by vector similarity. Returns structured hits.

    With `campsite_ids`, returns the closest `limit` claims *per campsite*;
    without it, the global closest `limit` overall.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    if campsite_ids is not None and not campsite_ids:
        return []
    vec_literal = embedding or _query_vec_literal(query)
    if campsite_ids is None:
        sql = _CLAIMS_GLOBAL_SQL
        params: tuple = (vec_literal, vec_literal, limit)
    else:
        sql = _CLAIMS_BY_SITE_SQL
        params = (
            [int(x) for x in campsite_ids],
            vec_literal,
            vec_literal,
            limit,
        )
    _attach_run_sql(sql, list(params))
    try:
        today = today_il()
        with stage("retrieve"):
            with connect(db_url) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        hits: list[dict] = []
        for campsite_id, claim_text, is_positive, published_at, distance in rows:
            day, days_ago = claim_recency(published_at, today=today)
            hits.append(
                {
                    "claim": claim_text or "N/A",
                    "campsite_id": campsite_id,
                    "is_positive": is_positive,
                    "date": day,
                    "days_ago": days_ago,
                    "distance": float(distance),
                }
            )
        return hits
    except Exception as e:
        return [{"error": f"Error searching claims: {e}"}]


def search_claims(query: str, limit: int = 5) -> str:
    """
    Search for review claims using vector similarity.

    Args:
        query: The search query (e.g., "fit for stargazing", "has hot water")
        limit: Maximum number of results to return (default: 5)

    Returns:
        A formatted string with matching claims, their campsite IDs, and relevance scores.
    """
    hits = search_review_claims(query, limit=limit)
    if not hits:
        return f"No claims found matching: {query}"
    if len(hits) == 1 and hits[0].get("error"):
        return str(hits[0]["error"])
    return "\n---\n".join(
        f"Campsite: {h.get('campsite_id')}\n"
        f"Claim: {h.get('claim')}\n"
        f"Date: {h.get('date')} ({h.get('days_ago')} days ago)\n"
        f"Relevance: {h.get('distance', 0):.4f}\n"
        for h in hits
    )


claims_search_tool = StructuredTool.from_function(
    func=search_claims,
    name="search_claims",
    description=(
        "Search for review claims about campsites using semantic similarity. "
        "Use this when users ask about specific features, amenities, or experiences "
        "at campsites (e.g., 'has hot water', 'good for stargazing', 'clean facilities') "
        "that are not numeric (like 'price < 100', 'rating > 4.5', 'distance < 100km', etc.). "
        "Returns matching claims with campsite IDs and relevance scores."
    ),
)
