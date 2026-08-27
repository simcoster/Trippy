"""
Fetch vacancies/prices from the INPA booking engine for campsites in Postgres.

Iterates the next N nights (default 14, one night each) and upserts into
`availability`, creating `accommodation_types` rows when a new room type appears.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import time
from datetime import date, timedelta
from html import unescape
from pathlib import Path
from urllib.parse import urlencode

import httpx
import psycopg
from amenity_enrichment import (
    amenity_llm_clients,
    enrich_accommodation_types,
    fill_missing_image_urls,
    load_types_with_amenities,
    LlmUsage,
    parse_room_categories,
)
from bs4 import BeautifulSoup
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

SCRAPER_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRAPER_DIR / "config.json"
RESULTS_PATH = "https://secure-hotels.net/INPA/BE_Results.aspx"

GET_OR_CREATE_ACCOMMODATION_SQL = """
INSERT INTO accommodation_types (hotel_id, name)
VALUES (%(hotel_id)s, %(name)s)
ON CONFLICT (hotel_id, name) DO UPDATE SET name = EXCLUDED.name
RETURNING id, name;
"""

UPSERT_AVAILABILITY_SQL = """
INSERT INTO availability (
    site_id, start_date, end_date, accommodation_type_id,
    price, adults_no, room_count
) VALUES (
    %(site_id)s, %(start_date)s, %(end_date)s,
    %(accommodation_type_id)s, %(price)s, %(adults_no)s, %(room_count)s
)
ON CONFLICT ON CONSTRAINT availability_unique_slot DO UPDATE
SET price = EXCLUDED.price,
    room_count = EXCLUDED.room_count,
    scraped_at = now()
RETURNING id;
"""

DELETE_AVAILABILITY_FOR_NIGHT_SQL = """
DELETE FROM availability
WHERE site_id = %(site_id)s
  AND start_date = %(start_date)s
  AND end_date = %(end_date)s
  AND adults_no = %(adults_no)s
"""

# Strip unit suffixes like "מספר 1" / "מספר 42" from room names.
ROOM_NUMBER_SUFFIX_RE = re.compile(r"\s*מספר\s+\d+\s*$")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


def database_url(config: dict) -> str:
    url = os.environ.get("DATABASE_URL") or config.get("database_url")
    if not url:
        raise RuntimeError("No database_url in config or DATABASE_URL env")
    return url.replace("@db:", "@localhost:")


def fetch_campsites(config: dict) -> list[dict]:
    """Load campsites that have a booking engine hotel id."""
    limit = int(config.get("availability", {}).get("limit_campsites", 2))
    sql = """
        SELECT id, name, booking_hotel_id
        FROM campsites
        WHERE booking_hotel_id IS NOT NULL
        ORDER BY id
        LIMIT %s
    """
    with psycopg.connect(database_url(config)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
    return [
        {"id": row[0], "name": row[1], "booking_hotel_id": row[2]}
        for row in rows
    ]


def search_url(
    hotel_id: str,
    check_in: date,
    check_out: date,
    *,
    rooms: int = 1,
    adults: int = 2,
    children: int = 0,
    infants: int = 0,
    lang: str = "heb",
) -> str:
    params = {
        "lang": lang,
        "hotel": hotel_id,
        "in": check_in.isoformat(),
        "out": check_out.isoformat(),
        "rooms": rooms,
        "ad1": adults,
        "ch1": children,
        "inf1": infants,
    }
    return f"{RESULTS_PATH}?{urlencode(params)}"


def parse_rooms(html: str) -> list[dict]:
    """
    Extract room type + price from BE_Results HTML.

    Preferred source: roomData="{...}" JSON on book buttons (structured).
    Fallback: .room-holder blocks with .roomname + .PriceD.
    """
    rooms: list[dict] = []
    seen: set[tuple] = set()

    for raw in re.findall(r'roomData="(\{.*?\})"', html):
        data = json.loads(unescape(raw).replace("&quot;", '"'))
        room_type = (data.get("RoomType") or data.get("PcName") or "").strip()
        price = data.get("Price")
        currency = (data.get("Currency") or "₪").strip()
        key = (room_type, price, data.get("RoomCode"), data.get("MatrixCode"))
        if key in seen:
            continue
        seen.add(key)
        rooms.append(
            {
                "room_type": room_type,
                "price": price,
                "currency": currency,
                "room_code": data.get("RoomCode"),
                "pc_name": (data.get("PcName") or "").strip() or None,
            }
        )

    if rooms:
        return rooms

    soup = BeautifulSoup(html, "html.parser")
    for holder in soup.select(".room-holder"):
        name_el = holder.select_one(".roomname")
        price_el = holder.select_one(".PriceD")
        if not name_el or not price_el:
            continue
        room_type = name_el.get_text(strip=True)
        price_raw = price_el.get("price") or price_el.get_text(strip=True)
        try:
            price = float(str(price_raw).replace(",", ""))
        except ValueError:
            price = price_raw
        key = (room_type, price)
        if key in seen:
            continue
        seen.add(key)
        rooms.append(
            {
                "room_type": room_type,
                "price": price,
                "currency": "₪",
                "room_code": None,
                "pc_name": None,
            }
        )
    return rooms


def fetch_results_html(url: str) -> str:
    with httpx.Client(
        timeout=45.0,
        verify=_ssl_context(),
        follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        },
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def fetch_availability(url: str) -> list[dict]:
    return parse_rooms(fetch_results_html(url))


def normalize_accommodation_name(name: str) -> str:
    """Drop 'מספר N' suffixes so numbered units share one accommodation type."""
    return ROOM_NUMBER_SUFFIX_RE.sub("", (name or "").strip()).strip()


def aggregate_offerings(offerings: list[dict]) -> list[dict]:
    """
    Group offerings by normalized room type.

    Example: 'בונגלו עם מזגן מספר 1' + '… מספר 3' → one row, room_count=2.
    Price is the minimum among instances of that type.
    """
    grouped: dict[str, dict] = {}
    for offer in offerings:
        raw_name = (offer.get("room_type") or "").strip()
        if not raw_name:
            continue
        price = offer.get("price")
        if price is None:
            continue
        name = normalize_accommodation_name(raw_name)
        if not name:
            continue
        price_f = float(price)
        if name not in grouped:
            grouped[name] = {
                "room_type": name,
                "price": price_f,
                "currency": offer.get("currency", "₪"),
                "room_count": 1,
            }
        else:
            grouped[name]["room_count"] += 1
            grouped[name]["price"] = min(grouped[name]["price"], price_f)
    return list(grouped.values())


def get_or_create_accommodation_type(cur, *, hotel_id: int, name: str) -> int:
    cur.execute(
        GET_OR_CREATE_ACCOMMODATION_SQL,
        {"hotel_id": hotel_id, "name": name},
    )
    row = cur.fetchone()
    return int(row[0])


def clear_availability_for_night(
    conn,
    *,
    site_id: int,
    start: date,
    end: date,
    adults_no: int,
) -> int:
    """Remove existing rows for this site/night/party size before re-scraping."""
    with conn.cursor() as cur:
        cur.execute(
            DELETE_AVAILABILITY_FOR_NIGHT_SQL,
            {
                "site_id": site_id,
                "start_date": start,
                "end_date": end,
                "adults_no": adults_no,
            },
        )
        return cur.rowcount


def upsert_availability_rows(
    conn,
    *,
    site_id: int,
    start: date,
    end: date,
    adults_no: int,
    offerings: list[dict],
) -> int:
    # Always replace this night's snapshot so removed room types don't linger.
    deleted = clear_availability_for_night(
        conn,
        site_id=site_id,
        start=start,
        end=end,
        adults_no=adults_no,
    )
    aggregated = aggregate_offerings(offerings)
    saved = 0
    with conn.cursor() as cur:
        for offer in aggregated:
            accom_id = get_or_create_accommodation_type(
                cur, hotel_id=site_id, name=offer["room_type"]
            )
            cur.execute(
                UPSERT_AVAILABILITY_SQL,
                {
                    "site_id": site_id,
                    "start_date": start,
                    "end_date": end,
                    "accommodation_type_id": accom_id,
                    "price": float(offer["price"]),
                    "adults_no": adults_no,
                    "room_count": int(offer["room_count"]),
                },
            )
            saved += 1
    if deleted:
        print(f"    cleared {deleted} existing row(s)")
    return saved


def night_windows(nights: int, start_from: date | None = None) -> list[tuple[date, date]]:
    """Return (check_in, check_out) pairs for `nights` consecutive one-night stays."""
    start = start_from or date.today()
    return [
        (start + timedelta(days=i), start + timedelta(days=i + 1))
        for i in range(nights)
    ]


def main() -> None:
    config = load_config()
    avail = config.get("availability", {})

    nights = int(avail.get("nights", 14))
    adults = int(avail.get("adults", 2))
    children = int(avail.get("children", 0))
    infants = int(avail.get("infants", 0))
    rooms_count = int(avail.get("rooms", 1))
    lang = avail.get("lang", "heb")
    pause_s = float(avail.get("request_pause_seconds", 0.5))

    campsites = fetch_campsites(config)
    if not campsites:
        print("No campsites with booking_hotel_id found")
        return

    windows = night_windows(nights)
    print(f"Scanning {len(windows)} nights starting {windows[0][0]} for {adults} adults")
    print(f"Campsites: {len(campsites)}")

    extractor, embedder = amenity_llm_clients()
    amenity_llm_usage = LlmUsage()

    total_saved = 0
    with psycopg.connect(database_url(config)) as conn:
        for site in campsites:
            print("=" * 60)
            print(f"{site['id']}. {site['name']}  ({site['booking_hotel_id']})")

            types_with_amenities = load_types_with_amenities(conn, site["id"])
            print(
                f"  accommodation types with amenities: "
                f"{len(types_with_amenities)} "
                f"({', '.join(sorted(types_with_amenities)) or 'none'})"
            )

            for check_in, check_out in windows:
                url = search_url(
                    site["booking_hotel_id"],
                    check_in,
                    check_out,
                    rooms=rooms_count,
                    adults=adults,
                    children=children,
                    infants=infants,
                    lang=lang,
                )
                print(f"  {check_in} → {check_out}")
                try:
                    html = fetch_results_html(url)
                except httpx.HTTPError as e:
                    print(f"    HTTP error: {e}")
                    continue

                offerings = parse_rooms(html)
                room_media = parse_room_categories(
                    html, normalize_accommodation_name
                )

                if not offerings:
                    print("    No room types returned")
                    deleted = clear_availability_for_night(
                        conn,
                        site_id=site["id"],
                        start=check_in,
                        end=check_out,
                        adults_no=adults,
                    )
                    conn.commit()
                    if deleted:
                        print(f"    cleared {deleted} existing row(s)")
                else:
                    aggregated = aggregate_offerings(offerings)
                    for offer in aggregated:
                        print(
                            f"    {offer['room_type']}  ×{offer['room_count']}  —  "
                            f"{offer.get('currency', '₪')}{offer['price']}"
                        )

                    missing_amenity_types = [
                        offer["room_type"]
                        for offer in aggregated
                        if offer["room_type"] not in types_with_amenities
                    ]
                    if missing_amenity_types:
                        print(
                            f"    enriching amenities for "
                            f"{len(missing_amenity_types)} type(s)"
                        )
                        newly = enrich_accommodation_types(
                            conn,
                            extractor,
                            embedder,
                            hotel_id=site["id"],
                            type_names=missing_amenity_types,
                            room_media=room_media,
                            get_or_create_type=get_or_create_accommodation_type,
                            usage=amenity_llm_usage,
                        )
                        types_with_amenities.update(newly)
                        conn.commit()

                    filled = fill_missing_image_urls(
                        conn,
                        hotel_id=site["id"],
                        room_media=room_media,
                    )
                    if filled:
                        conn.commit()
                        print(f"    filled image_urls on {filled} type(s)")

                    saved = upsert_availability_rows(
                        conn,
                        site_id=site["id"],
                        start=check_in,
                        end=check_out,
                        adults_no=adults,
                        offerings=offerings,
                    )
                    conn.commit()
                    total_saved += saved
                    print(f"    upserted {saved} row(s)")

                if pause_s > 0:
                    time.sleep(pause_s)

    print("-" * 60)
    print(f"Done. Upserted {total_saved} availability row(s).")
    if amenity_llm_usage.chat_calls or amenity_llm_usage.embed_calls:
        print(amenity_llm_usage.summary(prefix="Amenity enrich total: "))


if __name__ == "__main__":
    main()
