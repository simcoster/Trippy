"""
Discover booking-engine hotel (campsite) IDs from secure-hotels.net/INPA.

The availability search URL looks like:
  BE_Results.aspx?lang=heb&hotel=9_1&in=YYYY-MM-DD&out=YYYY-MM-DD&rooms=1&ad1=2&ch1=0&inf1=0

`hotel` is `{OptimaResortID}_{Wing}` (e.g. 9_1 for חורשת טל).

Writes discovered IDs into campsites.booking_hotel_id (matched by name).

Note: NoResults.aspx currently 404s. The hotel dropdown + settings JSON live on
https://secure-hotels.net/INPA/ (booking engine home), which is what this scrapes.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from datetime import date, timedelta
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path
from urllib.parse import urlencode

import httpx
import psycopg
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from source.scraper.tls import ssl_context

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

SCRAPER_DIR = Path(__file__).resolve().parent
ENGINE_HOME = "https://secure-hotels.net/INPA/"
RESULTS_PATH = "https://secure-hotels.net/INPA/BE_Results.aspx"
OUTPUT_FILE = SCRAPER_DIR / "booking_hotels.json"

ENSURE_COLUMN_SQL = """
ALTER TABLE campsites
ADD COLUMN IF NOT EXISTS booking_hotel_id TEXT;
"""

ENSURE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS campsites_booking_hotel_id_uidx
ON campsites (booking_hotel_id)
WHERE booking_hotel_id IS NOT NULL;
"""

UPDATE_BY_ID_SQL = """
UPDATE campsites
SET booking_hotel_id = %(booking_hotel_id)s
WHERE id = %(id)s
RETURNING id, name, booking_hotel_id;
"""


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        url = "postgresql://trippy:trippy@localhost:5432/trippy"
    return url.replace("@db:", "@localhost:")


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=30.0,
        verify=ssl_context(),
        follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        },
    )


def normalize_name(name: str) -> str:
    """Normalize Hebrew site names for fuzzy matching across sources."""
    text = unicodedata.normalize("NFKC", name or "")
    text = text.replace("–", " ").replace("—", " ").replace("-", " ")
    text = re.sub(r"[()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def core_name(name: str) -> str:
    """Strip shared marketing prefixes so cores can match across sources."""
    text = normalize_name(name)
    for token in ("חניון לילה", "גן לאומי", "שמורת טבע", "אתר"):
        text = text.replace(token, " ")
    return re.sub(r"\s+", " ", text).strip()


def names_match(a: str, b: str) -> bool:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True

    ca, cb = core_name(a), core_name(b)
    if not ca or not cb:
        return False
    if ca == cb or ca in cb or cb in ca:
        return True

    # Spelling variants (e.g. יטבתה vs יוטבתה)
    return SequenceMatcher(None, ca, cb).ratio() >= 0.82


def match_campsite(
    hotel_name: str, campsites: list[tuple[int, str]]
) -> tuple[int, str] | None:
    """Match a booking-engine name to a DB campsite row (id, name)."""
    matches = [
        (cid, name) for cid, name in campsites if names_match(hotel_name, name)
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    hotel_core = core_name(hotel_name)

    def score(row: tuple[int, str]) -> float:
        return SequenceMatcher(None, hotel_core, core_name(row[1])).ratio()

    matches.sort(key=score, reverse=True)
    return matches[0]


def parse_hotels_from_settings(html: str) -> list[dict]:
    """Parse the hidden #settings JSON (Regions -> Resorts)."""
    soup = BeautifulSoup(html, "html.parser")
    settings_el = soup.find("input", id="settings")
    if not settings_el or not settings_el.get("value"):
        return []

    raw = unescape(settings_el["value"])
    data = json.loads(raw)

    hotels: list[dict] = []
    for region in data.get("Regions", []):
        region_id = region.get("RegionID")
        for resort in region.get("Resorts", []):
            optima = str(resort.get("OptimaResortID", "")).strip()
            wing = str(resort.get("Wing", "1")).strip() or "1"
            if not optima:
                continue
            hotel_id = f"{optima}_{wing}"
            hotels.append(
                {
                    "name": (resort.get("ResortName") or "").strip(),
                    "hotel_id": hotel_id,
                    "optima_resort_id": optima,
                    "wing": wing,
                    "resort_id": str(resort.get("ResortID", "")).strip(),
                    "region_id": str(resort.get("RegionID") or region_id or "").strip(),
                }
            )
    return hotels


def parse_hotels_from_dropdown(html: str) -> list[dict]:
    """Fallback: parse #ddlHotels <option value="9_1:">Name</option>."""
    soup = BeautifulSoup(html, "html.parser")
    select = soup.find("select", id="ddlHotels")
    if not select:
        return []

    hotels: list[dict] = []
    for option in select.find_all("option"):
        raw = (option.get("value") or "").strip()
        name = option.get_text(strip=True)
        if not raw or raw == "0":
            continue
        hotel_id = raw.rstrip(":")
        hotels.append({"name": name, "hotel_id": hotel_id})
    return hotels


def discover_hotels(url: str = ENGINE_HOME) -> list[dict]:
    """Fetch booking engine page and return hotel id/name list."""
    with _client() as client:
        response = client.get(url)
        response.raise_for_status()
        html = response.text

    hotels = parse_hotels_from_settings(html)
    source = "settings"
    if not hotels:
        hotels = parse_hotels_from_dropdown(html)
        source = "ddlHotels"

    print(f"Fetched {url} ({len(html)} bytes)")
    print(f"Parsed {len(hotels)} hotels from #{source}")
    return hotels


def search_url(
    hotel_id: str,
    check_in: date,
    check_out: date,
    *,
    rooms: int = 1,
    adults: int = 1,
    children: int = 0,
    infants: int = 0,
    lang: str = "heb",
) -> str:
    """Build a BE_Results availability search URL for one hotel."""
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


def update_booking_hotel_ids(hotels: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Set campsites.booking_hotel_id from discovered hotels.
    Returns (updated_rows, unmatched_hotels).
    """
    db_url = _database_url()
    updated: list[dict] = []
    unmatched: list[dict] = []

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(ENSURE_COLUMN_SQL)
            cur.execute(ENSURE_INDEX_SQL)
            cur.execute("SELECT id, name FROM campsites")
            campsites = list(cur.fetchall())

            for hotel in hotels:
                match = match_campsite(hotel["name"], campsites)
                if not match:
                    unmatched.append(hotel)
                    continue
                campsite_id, campsite_name = match
                cur.execute(
                    UPDATE_BY_ID_SQL,
                    {
                        "id": campsite_id,
                        "booking_hotel_id": hotel["hotel_id"],
                    },
                )
                row = cur.fetchone()
                if row:
                    updated.append(
                        {
                            "id": row[0],
                            "name": row[1],
                            "booking_hotel_id": row[2],
                            "matched_as": campsite_name,
                        }
                    )
        conn.commit()

    return updated, unmatched


def main() -> None:
    print(f"Discovering hotels from: {ENGINE_HOME}")
    print("-" * 80)

    try:
        hotels = discover_hotels()
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
        return

    if not hotels:
        print("No hotels found")
        return

    check_in = date.today() + timedelta(days=1)
    check_out = check_in + timedelta(days=1)
    for hotel in hotels:
        hotel["sample_search_url"] = search_url(
            hotel["hotel_id"], check_in, check_out
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(hotels, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(hotels)} hotels to {OUTPUT_FILE}")

    updated, unmatched = update_booking_hotel_ids(hotels)
    print(f"Updated {len(updated)} campsites with booking_hotel_id")
    for row in updated:
        print(f"  {row['id']:>3}  {row['booking_hotel_id']:>6}  {row['name']}")

    if unmatched:
        print(f"\nUnmatched hotels ({len(unmatched)}):")
        for hotel in unmatched:
            print(f"  {hotel['hotel_id']:>6}  {hotel['name']}")

    print("\nSample search (first hotel):")
    print(f"  {hotels[0]['sample_search_url']}")


if __name__ == "__main__":
    main()
