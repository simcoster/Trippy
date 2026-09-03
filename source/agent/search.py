"""Catalog, availability, amenity, and claims search — not graph wiring."""

from __future__ import annotations

import os
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import psycopg
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pgvector.psycopg import register_vector

from db.models import SubjectCategory
from source.agent.constraints import claim_recency, today_il
from source.agent.dates import iso_day, parse_iso_day, stay_night_starts
from source.scraper.amenity_enrichment.llm import ClaimsEmbeddingLLMClient
from source.scraper.info_site.quote import quote_night
from source.scraper.info_site.schemas import RatePeriod

load_dotenv()

_claims_embedder = ClaimsEmbeddingLLMClient()

OPEN_SLOTS_LIMIT = 80
_LAST_OPEN_SLOTS_QUERY: dict[str, Any] | None = None

_NAMED_CAMPSITE_ALIASES = {
    "horshat tal": "חורשת טל",
    "horashat tal": "חורשת טל",
    "hurshat tal": "חורשת טל",
}


def _record_open_slots_query(record: dict[str, Any]) -> dict[str, Any]:
    global _LAST_OPEN_SLOTS_QUERY
    _LAST_OPEN_SLOTS_QUERY = record
    return record


def _open_slots_sql(
    *,
    date_range: dict,
    site_id: int | list[int] | None,
    party_size: int | None,
    limit: int,
) -> tuple[str, list[Any]] | tuple[None, str]:
    nights = stay_night_starts(date_range)
    if not nights:
        return None, "no_date"
    stay_start = nights[0]
    stay_end = nights[-1] + timedelta(days=1)
    clauses = [
        "a.start_date >= %s",
        "a.start_date < %s",
        "a.end_date = a.start_date + 1",
    ]
    params: list[Any] = [stay_start, stay_end]
    if isinstance(site_id, list):
        ids = [int(x) for x in site_id]
        if not ids:
            return None, "empty_site_ids"
        clauses.append("a.site_id = ANY(%s)")
        params.append(ids)
    elif site_id is not None:
        clauses.append("a.site_id = %s")
        params.append(int(site_id))
    if party_size is not None:
        clauses.append("(at.max_occupancy IS NULL OR at.max_occupancy >= %s)")
        params.append(int(party_size))
    params.append(len(nights))
    params.append(limit)
    sql = (
        "SELECT a.site_id, c.name, MIN(a.start_date), MAX(a.end_date),\n"
        "       MIN(a.room_count), at.id, at.name, at.max_occupancy\n"
        "FROM availability a\n"
        "JOIN accommodation_types at ON at.id = a.accommodation_type_id\n"
        "JOIN campsites c ON c.id = a.site_id\n"
        f"WHERE {' AND '.join(clauses)}\n"
        "GROUP BY a.site_id, c.name, at.id, at.name, at.max_occupancy\n"
        "HAVING COUNT(DISTINCT a.start_date) = %s\n"
        "ORDER BY MIN(a.start_date), at.id\n"
        "LIMIT %s"
    )
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


def _rate_period_for_stay(date_range: dict | None) -> RatePeriod:
    if not isinstance(date_range, dict):
        return "weekday"
    start = parse_iso_day(date_range.get("start"))
    if start is None:
        return "weekday"
    end = parse_iso_day(date_range.get("end")) or (start + timedelta(days=1))
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
        with psycopg.connect(db_url) as conn:
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
) -> float | None:
    if not rates:
        return None
    adults = party_size if party_size and party_size > 0 else 1
    try:
        return float(quote_night(rates, adults=adults, rate_period=rate_period))
    except ValueError:
        return None


def search_open_slots(
    *,
    date_range: dict | None = None,
    site_id: int | list[int] | None = None,
    party_size: int | None = None,
    numeric_constraints: list | None = None,
    limit: int = OPEN_SLOTS_LIMIT,
) -> list[dict]:
    """Catalog vacancies for a stay window, with list-price quotes.

    Availability is stored as one-night rows. A multi-night stay matches
    only when the type has a row for every night in [start, end). Party
    size uses accommodation max_occupancy (scrape is 1-adult). Price
    filters use quote_night against list_prices. Optional site_id narrows
    to a named park.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        _record_open_slots_query({"skipped": "no_database_url"})
        return []
    if not isinstance(date_range, dict) or not date_range.get("start"):
        _record_open_slots_query({"skipped": "no_date"})
        return []
    built = _open_slots_sql(
        date_range=date_range,
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
        "rate_period": _rate_period_for_stay(date_range),
    }
    _record_open_slots_query(query_record)
    try:
        with psycopg.connect(db_url) as conn:
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
        slots.append(
            {
                "campsite_id": int(row[0]),
                "campsite": row[1],
                "start": iso_day(row[2]),
                "end": iso_day(row[3]),
                "room_count": int(row[4]),
                "accommodation_type_id": int(row[5]),
                "accommodation_type": row[6],
                "max_occupancy": occupancy,
                "occupancy_unknown": occupancy is None,
            }
        )
    type_ids = list({int(s["accommodation_type_id"]) for s in slots})
    prices = _load_list_prices(type_ids)
    rate_period = _rate_period_for_stay(date_range)
    price_constraint = _price_per_night_constraint(numeric_constraints)
    quoted: list[dict] = []
    for slot in slots:
        price = _quote_slot_price(
            prices.get(int(slot["accommodation_type_id"])) or [],
            party_size=party_size,
            rate_period=rate_period,
        )
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


def _campsite_lookup_terms(name: str) -> list[str]:
    text = (name or "").strip()
    if not text:
        return []
    terms = [text]
    key = " ".join(text.lower().replace("-", " ").split())
    alias = _NAMED_CAMPSITE_ALIASES.get(key)
    if alias and alias not in terms:
        terms.append(alias)
    return terms


def lookup_campsite_by_name(name: str) -> list[dict]:
    """Resolve a user-named park to campsite id(s). Not a catalog dump."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    terms = _campsite_lookup_terms(name)
    if not terms:
        return []
    like_patterns = [f"%{term}%" for term in terms]
    sql = """
        SELECT id, name, booking_hotel_id
        FROM campsites
        WHERE name ILIKE ANY(%s)
        ORDER BY id
        LIMIT 5
    """
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (like_patterns,))
                rows = cur.fetchall()
        return [
            {
                "id": int(row[0]),
                "name": row[1],
                "hotel_id": int(row[0]),
                "booking_hotel_id": row[2],
            }
            for row in rows
        ]
    except Exception as e:
        return [{"error": f"Error looking up campsite: {e}"}]


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
        with psycopg.connect(db_url) as conn:
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
    embedding = _claims_embedder.embed([query])[0]
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


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
    try:
        with psycopg.connect(db_url) as conn:
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


def search_site_amenities(
    query: str,
    limit: int = 5,
    *,
    embedding: str | None = None,
    campsite_ids: list[int] | None = None,
) -> list[dict]:
    """Rank campsites by closest site-wide (communal) amenity embedding."""
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
          ON cr.campsite_id = c.id AND cr.accommodation_type_id IS NULL
        JOIN subject_vectors a ON a.id = cr.subject_id
        WHERE {' AND '.join(clauses)}
        GROUP BY c.id, c.name
        ORDER BY distance
        LIMIT %s
    """
    try:
        with psycopg.connect(db_url) as conn:
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


# Global top-K over every claim. Used when no campsite scope is given.
_CLAIMS_GLOBAL_SQL = """
        SELECT c.campsite_id, c.claim, c.is_positive, r.published_at,
               c.embedding <#> %s::vector AS distance
        FROM claims c
        JOIN reviews r ON r.id = c.review_id
        WHERE c.claim IS NOT NULL
          AND r.skip_reason IS NULL
        ORDER BY c.embedding <#> %s::vector
        LIMIT %s
"""

# Top-`limit` per campsite. A global top-K crowds out a candidate site's best
# claim with another site's, so drive one scan per site instead of trimming
# after. LATERAL (not a window over the whole table) so each sub-select rides
# claim_campsite_idx.
_CLAIMS_BY_SITE_SQL = """
        SELECT s.campsite_id, x.claim, x.is_positive, x.published_at, x.distance
        FROM unnest(%s::bigint[]) AS s(campsite_id)
        CROSS JOIN LATERAL (
            SELECT c.claim, c.is_positive, r.published_at,
                   c.embedding <#> %s::vector AS distance
            FROM claims c
            JOIN reviews r ON r.id = c.review_id
            WHERE c.campsite_id = s.campsite_id
              AND c.claim IS NOT NULL
              AND c.embedding IS NOT NULL
              AND r.skip_reason IS NULL
            ORDER BY c.embedding <#> %s::vector
            LIMIT %s
        ) AS x
        ORDER BY x.distance
"""


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
    try:
        today = today_il()
        with psycopg.connect(db_url) as conn:
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
