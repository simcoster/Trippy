"""INPA booking-results URLs for a vacant stay.

The parks.org.il iframe mints an ASP.NET session; those tokens are not a
link. `BE_Results.aspx` with hotel, dates, and party is a public GET —
the same shape `populate_availability.search_url` already scrapes.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlencode

from db.connect import connect
from source.agent.constraints import party_size_from_numeric
from source.agent.dates import iso_day

RESULTS_PATH = "https://secure-hotels.net/INPA/BE_Results.aspx"

_HOTEL_IDS_SQL = """
SELECT c.id, COALESCE(c.booking_hotel_id, p.booking_hotel_id)
FROM campsites c
LEFT JOIN campsites p ON p.id = c.parent_id
WHERE c.id = ANY(%s)
"""


def booking_results_url(
    hotel_id: str | None,
    check_in: Any,
    check_out: Any,
    *,
    adults: int = 1,
    children: int = 0,
    infants: int = 0,
    rooms: int = 1,
    lang: str = "heb",
) -> str | None:
    """Search-results URL for one hotel stay. None if hotel or dates are missing."""
    hotel = str(hotel_id or "").strip()
    start = iso_day(check_in).strip() if check_in is not None else ""
    end = iso_day(check_out).strip() if check_out is not None else ""
    if not hotel or not start or not end:
        return None
    adults_n = adults if isinstance(adults, int) and adults > 0 else 1
    params = {
        "lang": lang,
        "hotel": hotel,
        "in": start,
        "out": end,
        "rooms": rooms if isinstance(rooms, int) and rooms > 0 else 1,
        "ad1": adults_n,
        "ch1": children if isinstance(children, int) and children > 0 else 0,
        "inf1": infants if isinstance(infants, int) and infants > 0 else 0,
    }
    return f"{RESULTS_PATH}?{urlencode(params)}"


def booking_adults(constraints: dict[str, Any] | None) -> int:
    numeric = (constraints or {}).get("numeric_constraints") or []
    size = party_size_from_numeric(numeric)
    if size is not None and size > 0:
        return size
    return 1


def booking_hotel_ids_for_sites(site_ids: list[int]) -> dict[int, str]:
    """`campsites.booking_hotel_id`, falling back to the parent on a subcamp."""
    ids = list(dict.fromkeys(int(sid) for sid in site_ids))
    if not ids or not os.environ.get("DATABASE_URL"):
        return {}
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_HOTEL_IDS_SQL, (ids,))
                rows = cur.fetchall()
    except Exception:
        return {}
    out: dict[int, str] = {}
    for site_id, hotel in rows:
        text = str(hotel or "").strip()
        if text:
            out[int(site_id)] = text
    return out


def attach_booking_urls(
    fits: list[dict[str, Any]],
    constraints: dict[str, Any] | None,
    *,
    hotel_ids: dict[int, str] | None = None,
) -> None:
    """Set `booking_url` on each fit that has a hotel id and stay dates."""
    adults = booking_adults(constraints)
    missing = [
        int(fit["campsite_id"])
        for fit in fits
        if isinstance(fit, dict)
        and fit.get("campsite_id") is not None
        and not str(fit.get("booking_hotel_id") or "").strip()
    ]
    hotels = hotel_ids if hotel_ids is not None else booking_hotel_ids_for_sites(missing)
    for fit in fits:
        if not isinstance(fit, dict):
            continue
        hotel = str(fit.get("booking_hotel_id") or "").strip()
        if not hotel:
            cid = fit.get("campsite_id")
            hotel = hotels.get(int(cid), "") if cid is not None else ""
        url = booking_results_url(
            hotel or None,
            fit.get("start"),
            fit.get("end"),
            adults=adults,
        )
        if url:
            fit["booking_url"] = url
            if hotel:
                fit["booking_hotel_id"] = hotel
