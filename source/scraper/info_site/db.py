"""Postgres upserts for info-site lodging names and list prices."""

from __future__ import annotations

from .schemas import ClassifiedPriceRow

GET_OR_CREATE_INFO_WEBSITE_NAME_SQL = """
INSERT INTO info_website_names (site_id, name)
VALUES (%(site_id)s, %(name)s)
ON CONFLICT (site_id, name) DO UPDATE SET name = EXCLUDED.name
RETURNING id, name;
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


def get_or_create_info_website_name(cur, *, site_id: int, name: str) -> int:
    cur.execute(
        GET_OR_CREATE_INFO_WEBSITE_NAME_SQL,
        {"site_id": site_id, "name": name},
    )
    row = cur.fetchone()
    return int(row[0])


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
) -> list[ClassifiedPriceRow]:
    """Replace regular list prices for a site. Persists lodging rows only."""
    from .classify import lodging_rows_to_persist

    lodging = lodging_rows_to_persist(rows)
    with conn.cursor() as cur:
        cur.execute(
            DELETE_REGULAR_LIST_PRICES_SQL,
            {"site_id": site_id, "rate_class": rate_class},
        )
        for row in lodging:
            name_id = get_or_create_info_website_name(
                cur, site_id=site_id, name=row.accommodation_type
            )
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
    return lodging
