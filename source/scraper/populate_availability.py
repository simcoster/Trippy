"""
Fetch vacancies/prices from the INPA booking engine for campsites in Postgres.

For now: load the first campsite with a booking_hotel_id, query one night
(defaults: 2026-08-25 → 2026-08-26, 2 adults), print room type + price.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
from datetime import date
from html import unescape
from pathlib import Path
from urllib.parse import urlencode

import httpx
import psycopg
from bs4 import BeautifulSoup
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

SCRAPER_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRAPER_DIR / "config.json"
RESULTS_PATH = "https://secure-hotels.net/INPA/BE_Results.aspx"


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
    limit = int(config.get("availability", {}).get("limit_campsites", 1))
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


def fetch_availability(url: str) -> list[dict]:
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
        return parse_rooms(response.text)


def main() -> None:
    config = load_config()
    avail = config.get("availability", {})

    check_in = date.fromisoformat(avail.get("check_in", "2026-08-25"))
    check_out = date.fromisoformat(avail.get("check_out", "2026-08-26"))
    adults = int(avail.get("adults", 2))
    children = int(avail.get("children", 0))
    infants = int(avail.get("infants", 0))
    rooms_count = int(avail.get("rooms", 1))
    lang = avail.get("lang", "heb")

    campsites = fetch_campsites(config)
    if not campsites:
        print("No campsites with booking_hotel_id found")
        return

    for site in campsites:
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
        print(f"{site['id']}. {site['name']}  ({site['booking_hotel_id']})")
        print(f"   {check_in} → {check_out}, {adults} adults")
        print(f"   {url}")
        print("-" * 60)

        try:
            offerings = fetch_availability(url)
        except httpx.HTTPError as e:
            print(f"   HTTP error: {e}")
            continue

        if not offerings:
            print("   No room types returned")
            continue

        for offer in offerings:
            price = offer["price"]
            currency = offer["currency"]
            print(f"   {offer['room_type']}  —  {currency}{price}")


if __name__ == "__main__":
    main()
