"""Postgres writes for info-site list prices.

`info_website_names` rows are created by `scrape-rooms` off the lodging panel,
which is the campsite's catalog. This scrape *finds* the row a rate-card label
belongs to and attaches a price to it, the same way the availability scrape
finds a type for a booking name. A label matching nothing is skipped and
reported rather than inventing a lodging product the operator never listed.
"""

from __future__ import annotations

from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.info_site.match_listing import (
    InfoWebsiteNameMatcher,
    match_info_website_name,
)

from .schemas import ClassifiedPriceRow

LOAD_INFO_WEBSITE_NAMES_SQL = """
SELECT id, name FROM info_website_names WHERE site_id = %(site_id)s ORDER BY id
"""

DELETE_REGULAR_LIST_PRICES_SQL = """
DELETE FROM list_prices
WHERE site_id = %(site_id)s
  AND rate_class = %(rate_class)s
"""

INSERT_LIST_PRICE_SQL = """
INSERT INTO list_prices (
    site_id, info_website_name_id, guest_type, rate_period, rate_class,
    price, currency, notes, raw_label
) VALUES (
    %(site_id)s, %(info_website_name_id)s, %(guest_type)s, %(rate_period)s,
    %(rate_class)s, %(price)s, %(currency)s, %(notes)s, %(raw_label)s
)
ON CONFLICT ON CONSTRAINT list_prices_unique_rate DO UPDATE
SET price = EXCLUDED.price,
    currency = EXCLUDED.currency,
    notes = EXCLUDED.notes,
    raw_label = EXCLUDED.raw_label,
    scraped_at = now(),
    updated_at = now()
RETURNING id;
"""

FILL_BOOKING_HOTEL_ID_SQL = """
UPDATE campsites
SET booking_hotel_id = %(booking_hotel_id)s
WHERE id = %(id)s
  AND booking_hotel_id IS NULL
RETURNING id, booking_hotel_id;
"""


CREATE_INFO_WEBSITE_NAME_SQL = """
INSERT INTO info_website_names (site_id, name)
VALUES (%(site_id)s, %(name)s)
ON CONFLICT (site_id, name) DO UPDATE SET name = EXCLUDED.name
RETURNING id
"""


def get_or_create_info_website_name(cur, *, site_id: int, name: str) -> int:
    """The lodging product row for a name, created if new.

    Called by `scrape-rooms` off the lodging panel headings, and by nothing
    else: the panel is the catalog, so a name that is not on it is not a
    product. The price scrape matches against what this created.
    """
    cur.execute(
        CREATE_INFO_WEBSITE_NAME_SQL,
        {"site_id": site_id, "name": name},
    )
    return int(cur.fetchone()[0])


def maybe_fill_booking_hotel_id(
    cur, *, site_id: int, booking_hotel_id: str | None
) -> str | None:
    if not booking_hotel_id:
        return None
    cur.execute(
        FILL_BOOKING_HOTEL_ID_SQL,
        {"id": site_id, "booking_hotel_id": booking_hotel_id},
    )
    row = cur.fetchone()
    return row[1] if row else None


def snapshot_list_prices(
    conn,
    *,
    site_id: int,
    rows: list[ClassifiedPriceRow],
    rate_class: str = "regular",
    currency: str = "ILS",
    matcher: InfoWebsiteNameMatcher | None = None,
    usage: LlmUsage | None = None,
    unmatched_sink: list[str] | None = None,
) -> list[ClassifiedPriceRow]:
    """Replace regular list prices for a site. Persists lodging rows only.

    Each rate-card label is resolved to an existing `info_website_names` row --
    exact name, else one 30B pick over the names `scrape-rooms` created. A label
    that resolves to nothing is skipped: the rate card sometimes prices things
    the lodging panel does not list, and a price with no product is not
    something the planner can use.
    """
    from .classify import lodging_rows_to_persist

    lodging = lodging_rows_to_persist(rows)
    with conn.cursor() as cur:
        cur.execute(LOAD_INFO_WEBSITE_NAMES_SQL, {"site_id": site_id})
        names = [(int(r[0]), r[1]) for r in cur.fetchall()]
        cur.execute(
            DELETE_REGULAR_LIST_PRICES_SQL,
            {"site_id": site_id, "rate_class": rate_class},
        )
        stored: list[ClassifiedPriceRow] = []
        for row in lodging:
            name_id = match_info_website_name(
                row.accommodation_type, names, matcher=matcher, usage=usage
            )
            if name_id is None:
                print(f"      NO LODGING MATCH, price skipped: {row.raw_label!r}")
                if unmatched_sink is not None:
                    unmatched_sink.append(row.accommodation_type)
                continue
            stored.append(row)
            cur.execute(
                INSERT_LIST_PRICE_SQL,
                {
                    "site_id": site_id,
                    "info_website_name_id": name_id,
                    "guest_type": row.guest_type,
                    "rate_period": row.rate_period,
                    "rate_class": rate_class,
                    "price": row.price,
                    "currency": currency,
                    "notes": row.notes,
                    "raw_label": row.raw_label,
                },
            )
    return stored
