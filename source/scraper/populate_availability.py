"""
Fetch vacancies from the INPA booking engine for campsites in Postgres.

Iterates the next N nights (default 28, one night each) and upserts into
`availability`. Creates accommodation_types from INPA names and links each
to an info_website_names row (exact name, else Qwen 30B).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from urllib.parse import urlencode

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from db.connect import connect
from source.agent.dates import today_il
from source.scraper.amenity_enrichment import (
    LlmUsage,
    fill_missing_image_urls,
    parse_room_categories,
)
from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_30B_MODEL,
    record_scrape_cost,
)
from source.scraper.availability_report import (
    AvailabilityRun,
    HttpError,
    LayoutSuspicion,
    NightCountKey,
    PageFingerprint,
    PastNightsDeleted,
    diff_night_counts,
    html_sha256,
    layout_suspicion,
    offers_sha256,
    render_run_report,
    should_skip_write,
    write_run_report,
)

# aliased: `site_ids` is also a local here, the subcamp id list.
from source.scraper.cli import site_ids as parse_site_ids
from source.scraper.info_site.match_listing import InfoWebsiteNameMatcher
from source.scraper.rules_ingest.subcamps import (
    Subcamp,
    group_by_owner,
    load_subcamps,
    owned_site_ids,
    unit_owner,
)
from source.scraper.tls import ssl_context

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

SCRAPER_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRAPER_DIR / "config.json"
RESULTS_PATH = "https://secure-hotels.net/INPA/BE_Results.aspx"

LOAD_INFO_WEBSITE_NAMES_SQL = """
SELECT id, name FROM info_website_names WHERE site_id = %(site_id)s
ORDER BY id
"""



UPSERT_AVAILABILITY_SQL = """
INSERT INTO availability (
    site_id, start_date, end_date, accommodation_type_id,
    room_count
) VALUES (
    %(site_id)s, %(start_date)s, %(end_date)s,
    %(accommodation_type_id)s, %(room_count)s
)
ON CONFLICT ON CONSTRAINT availability_unique_slot DO UPDATE
SET room_count = EXCLUDED.room_count,
    scraped_at = now(),
    updated_at = now()
RETURNING id;
"""

# `site_ids`, not `site_id`: a split site's units are owned by its subcamp rows,
# so re-scraping a night has to clear the parent and every child together.
DELETE_AVAILABILITY_FOR_NIGHT_SQL = """
DELETE FROM availability
WHERE site_id = ANY(%(site_ids)s)
  AND start_date = %(start_date)s
  AND end_date = %(end_date)s
"""

LOAD_PAGE_HASH_SQL = """
SELECT html_sha256, offers_sha256
FROM booking_page_hashes
WHERE site_id = %(site_id)s
  AND start_date = %(start_date)s
  AND end_date = %(end_date)s
"""

UPSERT_PAGE_HASH_SQL = """
INSERT INTO booking_page_hashes (
    site_id, start_date, end_date, html_sha256, offers_sha256
) VALUES (
    %(site_id)s, %(start_date)s, %(end_date)s,
    %(html_sha256)s, %(offers_sha256)s
)
ON CONFLICT ON CONSTRAINT booking_page_hashes_slot_key DO UPDATE
SET html_sha256 = EXCLUDED.html_sha256,
    offers_sha256 = EXCLUDED.offers_sha256,
    scraped_at = now(),
    updated_at = now()
"""

TOUCH_AVAILABILITY_SQL = """
UPDATE availability
SET scraped_at = now()
WHERE site_id = ANY(%(site_ids)s)
  AND start_date = %(start_date)s
  AND end_date = %(end_date)s
"""

TOUCH_PAGE_HASH_SQL = """
UPDATE booking_page_hashes
SET html_sha256 = %(html_sha256)s,
    scraped_at = now()
WHERE site_id = %(site_id)s
  AND start_date = %(start_date)s
  AND end_date = %(end_date)s
"""

DELETE_AVAILABILITY_BEFORE_SQL = """
DELETE FROM availability
WHERE start_date < %(before)s
"""

DELETE_PAGE_HASHES_BEFORE_SQL = """
DELETE FROM booking_page_hashes
WHERE start_date < %(before)s
"""

LOAD_NIGHT_COUNTS_SQL = """
SELECT a.site_id, t.name, a.room_count
FROM availability a
JOIN accommodation_types t ON t.id = a.accommodation_type_id
WHERE a.site_id = ANY(%(site_ids)s)
  AND a.start_date = %(start_date)s
  AND a.end_date = %(end_date)s
ORDER BY a.site_id, t.name
"""

# Strip unit suffixes: "מספר 1", "מספר 1-4", or a trailing unit number ("01", "15").
ROOM_NUMBER_SUFFIX_RE = re.compile(r"\s*(?:מספר\s+)?\d+(?:\s*-\s*\d+)?\s*$")
_WS_RE = re.compile(r"\s+")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def database_url(config: dict) -> str:
    url = os.environ.get("DATABASE_URL") or config.get("database_url")
    if not url:
        raise RuntimeError("No database_url in config or DATABASE_URL env")
    return url.replace("@db:", "@localhost:")


def fetch_campsites(config: dict, *, sites: list[int] | None = None) -> list[dict]:
    """Campsites with a booking engine hotel id, or just the ones `sites` names.

    Subcamps carry no `booking_hotel_id` — a split site has one booking id and
    one flat unit list, which `unit_owner` divides afterwards — so they never
    appear here and are never scraped on their own.
    """
    limit = int(config.get("availability", {}).get("limit_campsites", 2))
    where = "booking_hotel_id IS NOT NULL"
    params: list = []
    if sites:
        where += " AND id = ANY(%s)"
        params.append(list(sites))
        limit = len(sites)
    params.append(limit)
    sql = f"""
        SELECT id, name, booking_hotel_id
        FROM campsites
        WHERE {where}
        ORDER BY id
        LIMIT %s
    """
    with connect(database_url(config)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
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
        verify=ssl_context(),
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
    """Drop trailing unit numbers so numbered units share one accommodation type."""
    stripped = ROOM_NUMBER_SUFFIX_RE.sub("", (name or "").strip()).strip()
    return _WS_RE.sub(" ", stripped) if stripped else ""


def aggregate_offerings(offerings: list[dict]) -> list[dict]:
    """
    Group offerings by normalized room type.

    Example: 'בונגלו עם מזגן מספר 1' + '… מספר 3' → one row, room_count=2.
    """
    grouped: dict[str, dict] = {}
    for offer in offerings:
        raw_name = (offer.get("room_type") or "").strip()
        if not raw_name:
            continue
        name = normalize_accommodation_name(raw_name)
        if not name:
            continue
        if name not in grouped:
            grouped[name] = {
                "room_type": name,
                "room_count": 1,
            }
        else:
            grouped[name]["room_count"] += 1
    return list(grouped.values())


UNIT_MATCH_PROMPT = """You match a booking-engine lodging name to one of a campsite's known accommodation types.

Output valid JSON only, no markdown:
{"name": string | null}

Rules:
- "name" must be copied exactly from the provided type list, or null.
- Pick the same lodging product. A unit number, a room number or an inventory
  count is not part of the product: `בונגלו עם מזגן מספר 42` is `בונגלו עם מזגן`.
- Accessible and non-accessible are DIFFERENT products, and so are single and
  double: never match `חושה` to `חושה מונגשת` or to `חושה כפולה`.
- If none of the types is that product, return null. Never invent a name.
"""

LOOKUP_TYPE_SQL = """
SELECT id FROM accommodation_types
WHERE hotel_id = %(hotel_id)s AND (name = %(name)s OR aliases @> ARRAY[%(name)s])
"""

APPEND_ALIAS_SQL = """
UPDATE accommodation_types
SET aliases = array_append(aliases, %(alias)s), updated_at = now()
WHERE id = %(id)s AND NOT (aliases @> ARRAY[%(alias)s])
"""


def load_site_types(cur, hotel_ids: list[int]) -> list[tuple[int, str]]:
    """Every accommodation type on this site, subcamps included."""
    cur.execute(
        "SELECT id, name FROM accommodation_types WHERE hotel_id = ANY(%s) ORDER BY id",
        (hotel_ids,),
    )
    return [(int(r[0]), r[1]) for r in cur.fetchall()]


def match_accommodation_type(
    cur,
    *,
    hotel_id: int,
    booking_name: str,
    candidates: list[tuple[int, str]],
    matcher: InfoWebsiteNameMatcher | None = None,
    usage: LlmUsage | None = None,
) -> int | None:
    """A booking unit name -> the id of an already-scraped type, or None.

    Exact name, then the alias list, then one small LLM pick over this site's
    type names -- and a pick is remembered as an alias, so the same booking
    name is free ever after. Types come from the info page's lodging panel
    (`scrape-rooms`); this scrape creates none, because the booking engine
    shows only what is free tonight and would invent a type for a unit the
    operator never listed.
    """
    needle = normalize_accommodation_name(booking_name)
    if not needle:
        return None
    cur.execute(LOOKUP_TYPE_SQL, {"hotel_id": hotel_id, "name": needle})
    row = cur.fetchone()
    if row is not None:
        return int(row[0])
    if not candidates:
        return None
    # Availability may still skip: a booking unit the lodging panel never
    # listed is a real thing, and inventing a type for it loses the catalog.
    # Lowercase the 30B view: `מתחם pitch` vs catalog `מתחם PITCH` is the
    # same product, but the pick must copy a candidate exactly and otherwise
    # comes back null. Restore the original casing before storing the alias.
    original_by_folded = {name.lower(): name for _, name in candidates}
    picked, _confidence = (matcher or InfoWebsiteNameMatcher()).pick_name(
        needle.lower(),
        [name.lower() for _, name in candidates],
        usage=usage,
    )
    if picked is None:
        return None
    original = original_by_folded.get(picked.lower())
    if original is None:
        return None
    for type_id, name in candidates:
        if name == original:
            cur.execute(APPEND_ALIAS_SQL, {"id": type_id, "alias": needle})
            print(f"      alias {needle!r} -> {name!r}")
            return type_id
    return None


def clear_availability_for_night(
    conn,
    *,
    site_ids: list[int],
    start: date,
    end: date,
) -> int:
    """Remove existing rows for this site/night before re-scraping."""
    with conn.cursor() as cur:
        cur.execute(
            DELETE_AVAILABILITY_FOR_NIGHT_SQL,
            {
                "site_ids": list(site_ids),
                "start_date": start,
                "end_date": end,
            },
        )
        return cur.rowcount


def delete_past_nights(conn, *, before: date) -> PastNightsDeleted:
    """Drop availability (and page hashes) whose check-in is before `before`.

    The rolling window only covers today onward; leftover nights from
    earlier scrapes would otherwise sit in search forever.
    """
    with conn.cursor() as cur:
        cur.execute(DELETE_AVAILABILITY_BEFORE_SQL, {"before": before})
        availability_rows = cur.rowcount
        cur.execute(DELETE_PAGE_HASHES_BEFORE_SQL, {"before": before})
        hash_rows = cur.rowcount
    return PastNightsDeleted(
        availability_rows=availability_rows, hash_rows=hash_rows
    )


def load_page_hash(
    conn,
    *,
    site_id: int,
    start: date,
    end: date,
) -> PageFingerprint | None:
    with conn.cursor() as cur:
        cur.execute(
            LOAD_PAGE_HASH_SQL,
            {
                "site_id": site_id,
                "start_date": start,
                "end_date": end,
            },
        )
        row = cur.fetchone()
    if row is None:
        return None
    return PageFingerprint(html_sha256=str(row[0]), offers_sha256=str(row[1]))


def upsert_page_hash(
    conn,
    *,
    site_id: int,
    start: date,
    end: date,
    html_digest: str,
    offers_digest: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            UPSERT_PAGE_HASH_SQL,
            {
                "site_id": site_id,
                "start_date": start,
                "end_date": end,
                "html_sha256": html_digest,
                "offers_sha256": offers_digest,
            },
        )


def touch_availability_scraped_at(
    conn,
    *,
    site_ids: list[int],
    start: date,
    end: date,
) -> None:
    """Mark this night as checked without claiming vacancy changed."""
    with conn.cursor() as cur:
        cur.execute(
            TOUCH_AVAILABILITY_SQL,
            {
                "site_ids": list(site_ids),
                "start_date": start,
                "end_date": end,
            },
        )


def touch_page_hash(
    conn,
    *,
    site_id: int,
    start: date,
    end: date,
    html_digest: str,
) -> None:
    """Chrome may move; offers did not. Leave updated_at alone."""
    with conn.cursor() as cur:
        cur.execute(
            TOUCH_PAGE_HASH_SQL,
            {
                "site_id": site_id,
                "start_date": start,
                "end_date": end,
                "html_sha256": html_digest,
            },
        )


def load_night_counts(
    conn,
    *,
    site_ids: list[int],
    start: date,
    end: date,
) -> dict[NightCountKey, int]:
    with conn.cursor() as cur:
        cur.execute(
            LOAD_NIGHT_COUNTS_SQL,
            {
                "site_ids": list(site_ids),
                "start_date": start,
                "end_date": end,
            },
        )
        rows = cur.fetchall()
    return {
        NightCountKey(site_id=int(site_id), type_name=name): int(count)
        for site_id, name, count in rows
    }


def upsert_availability_rows(
    conn,
    *,
    site_id: int,
    start: date,
    end: date,
    offerings: list[dict],
    matcher: InfoWebsiteNameMatcher | None = None,
    usage: LlmUsage | None = None,
    subcamps: list[Subcamp] | None = None,
    unmatched_sink: list[tuple[int, str]] | None = None,
) -> int:
    """Replace one night's snapshot for a site, routing units to their subcamp.

    The booking engine has one hotel id for a split site, so every row it
    returns arrives under the parent. `unit_owner` decides which campsite row
    each unit type actually belongs to; for an ordinary site that is always the
    parent, and this reads exactly as it did before.
    """
    subcamps = list(subcamps or ())
    # Always replace this night's snapshot so removed room types don't linger.
    deleted = clear_availability_for_night(
        conn,
        site_ids=owned_site_ids(site_id, subcamps),
        start=start,
        end=end,
    )
    aggregated = aggregate_offerings(offerings)
    saved = 0
    with conn.cursor() as cur:
        candidates = load_site_types(cur, owned_site_ids(site_id, subcamps))
        for offer in aggregated:
            owner = unit_owner(offer["room_type"], site_id, subcamps)
            accom_id = match_accommodation_type(
                cur,
                hotel_id=owner,
                booking_name=offer["room_type"],
                candidates=candidates,
                matcher=matcher,
                usage=usage,
            )
            if accom_id is None:
                # The lodging panel does not list this unit. Skipping is the
                # point: the info page is the catalog, and inventing a type
                # here is what `scrape-rooms` exists to stop.
                print(f"      NO TYPE MATCH, offer skipped: {offer['room_type']!r}")
                if unmatched_sink is not None:
                    unmatched_sink.append((owner, offer["room_type"]))
                continue
            cur.execute(
                UPSERT_AVAILABILITY_SQL,
                {
                    "site_id": owner,
                    "start_date": start,
                    "end_date": end,
                    "accommodation_type_id": accom_id,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="INPA availability scraper")
    parser.add_argument(
        "--site",
        action="append",
        default=None,
        metavar="ID[,ID...]",
        help="Campsite ids to scrape (a parent id for a split site); "
        "repeat or comma-separate. Default: all",
    )
    args = parser.parse_args(argv)

    config = load_config()
    avail = config.get("availability", {})

    nights = int(avail.get("nights", 28))
    adults = int(avail.get("adults", 2))
    children = int(avail.get("children", 0))
    infants = int(avail.get("infants", 0))
    rooms_count = int(avail.get("rooms", 1))
    lang = avail.get("lang", "heb")
    pause_s = float(avail.get("request_pause_seconds", 0.5))

    campsites = fetch_campsites(config, sites=parse_site_ids(args.site))
    if not campsites:
        print("No campsites with booking_hotel_id found")
        return

    today = today_il()
    windows = night_windows(nights, start_from=today)
    print(f"Scanning {len(windows)} nights starting {windows[0][0]} for {adults} adults")
    print(f"Campsites: {len(campsites)}")

    listing_llm_usage = LlmUsage()
    # Booking names are matched onto the types `scrape-rooms` created, never
    # used to create one. A non-exact match is remembered as an alias.
    # Pinned to the 30B: the move to the 235B was measured on the rate-card
    # listing prompt, and this is a different prompt doing a different job.
    type_matcher = InfoWebsiteNameMatcher(
        system_prompt=UNIT_MATCH_PROMPT,
        role="unit_match",
        model=QWEN_INSTRUCT_30B_MODEL,
    )
    unmatched: list[tuple[int, str]] = []
    started_at = datetime.now(timezone.utc)
    started_mono = time.monotonic()
    run = AvailabilityRun(
        started_at=started_at,
        sites=len(campsites),
        nights=len(windows),
    )

    total_saved = 0
    with connect(database_url(config)) as conn:
        pruned = delete_past_nights(conn, before=today)
        conn.commit()
        run.past_rows_deleted = pruned.availability_rows
        run.past_hashes_deleted = pruned.hash_rows
        if pruned.availability_rows or pruned.hash_rows:
            print(
                f"Dropped {pruned.availability_rows} availability row(s) "
                f"and {pruned.hash_rows} page hash(es) before {today}"
            )

        for site in campsites:
            print("=" * 60)
            print(f"{site['id']}. {site['name']}  ({site['booking_hotel_id']})")

            subcamps = load_subcamps(conn, site["id"])
            site_ids = owned_site_ids(site["id"], subcamps)
            if subcamps:
                print(
                    f"  subcamps: {len(subcamps)} "
                    f"({', '.join(sub.heading for sub in subcamps)})"
                )
            with conn.cursor() as cur:
                site_types = load_site_types(cur, site_ids)
            print(
                f"  accommodation types: {len(site_types)} "
                f"({', '.join(name for _, name in site_types) or 'none'})"
            )
            if not site_types:
                print("  NO TYPES — run `just scrape-rooms` for this site first")

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
                    run.http_errors.append(
                        HttpError(
                            site_id=int(site["id"]),
                            site_name=site["name"],
                            start_date=check_in,
                            message=str(e),
                        )
                    )
                    continue

                run.pages_fetched += 1
                offerings = parse_rooms(html)
                aggregated = aggregate_offerings(offerings)
                html_digest = html_sha256(html)
                offers_digest = offers_sha256(aggregated)
                stored = load_page_hash(
                    conn,
                    site_id=int(site["id"]),
                    start=check_in,
                    end=check_out,
                )
                previous_offers = stored.offers_sha256 if stored is not None else None
                if layout_suspicion(
                    previous_offers_sha=previous_offers,
                    new_aggregated=aggregated,
                ):
                    run.layout_suspicions.append(
                        LayoutSuspicion(
                            site_id=int(site["id"]),
                            site_name=site["name"],
                            start_date=check_in,
                            previous_had_offers=previous_offers != offers_sha256([]),
                            now_empty=not aggregated,
                        )
                    )
                    print("    LAYOUT SUSPICION: empty vs previous offers mismatch")

                if should_skip_write(previous_offers, offers_digest):
                    touch_page_hash(
                        conn,
                        site_id=int(site["id"]),
                        start=check_in,
                        end=check_out,
                        html_digest=html_digest,
                    )
                    touch_availability_scraped_at(
                        conn,
                        site_ids=site_ids,
                        start=check_in,
                        end=check_out,
                    )
                    conn.commit()
                    run.pages_skipped += 1
                    print("    unchanged (offers hash) — skipped write")
                    if pause_s > 0:
                        time.sleep(pause_s)
                    continue

                room_media = parse_room_categories(
                    html, normalize_accommodation_name
                )
                old_counts = load_night_counts(
                    conn,
                    site_ids=site_ids,
                    start=check_in,
                    end=check_out,
                )

                if not offerings:
                    print("    No room types returned")
                    deleted = clear_availability_for_night(
                        conn,
                        site_ids=site_ids,
                        start=check_in,
                        end=check_out,
                    )
                    if deleted:
                        print(f"    cleared {deleted} existing row(s)")
                else:
                    for offer in aggregated:
                        owner = unit_owner(offer["room_type"], site["id"], subcamps)
                        tail = f"  → {owner}" if owner != site["id"] else ""
                        print(
                            f"    {offer['room_type']}  "
                            f"×{offer['room_count']}{tail}"
                        )

                    filled = 0
                    for owner, owned in group_by_owner(
                        room_media, site["id"], subcamps
                    ).items():
                        filled += fill_missing_image_urls(
                            conn,
                            hotel_id=owner,
                            room_media={name: room_media[name] for name in owned},
                        )
                    if filled:
                        print(f"    filled image_urls on {filled} type(s)")

                    saved = upsert_availability_rows(
                        conn,
                        site_id=site["id"],
                        start=check_in,
                        end=check_out,
                        offerings=offerings,
                        matcher=type_matcher,
                        usage=listing_llm_usage,
                        subcamps=subcamps,
                        unmatched_sink=unmatched,
                    )
                    total_saved += saved
                    print(f"    upserted {saved} row(s)")

                new_counts = load_night_counts(
                    conn,
                    site_ids=site_ids,
                    start=check_in,
                    end=check_out,
                )
                run.changes.extend(
                    diff_night_counts(
                        site_name=site["name"],
                        start=check_in,
                        old=old_counts,
                        new=new_counts,
                    )
                )
                upsert_page_hash(
                    conn,
                    site_id=int(site["id"]),
                    start=check_in,
                    end=check_out,
                    html_digest=html_digest,
                    offers_digest=offers_digest,
                )
                conn.commit()

                if pause_s > 0:
                    time.sleep(pause_s)

    run.rows_upserted = total_saved
    run.unmatched = unmatched
    run.seconds = time.monotonic() - started_mono

    print("-" * 60)
    print(f"Done. Upserted {total_saved} availability row(s).")
    if unmatched:
        print(f"{len(unmatched)} booking unit(s) matched no accommodation type:")
        for site_id, name in sorted(set(unmatched)):
            print(f"  campsite {site_id}: {name}")
    if listing_llm_usage.chat_calls:
        print(listing_llm_usage.summary(prefix="Info-site name match total: "))

    # One record for the run; the per-role rows keep name-matching, unit-detail
    # extraction, rule extraction, judging and embedding apart.
    run_usage = LlmUsage()
    run_usage.merge(listing_llm_usage)
    written = record_scrape_cost("scrape-availability", run_usage)
    if written:
        print(run_usage.summary(prefix="Scrape total: "))
        print(f"cost report appended to {written}")

    report_text = render_run_report(run, run_usage)
    print(report_text)
    report_path = write_run_report(report_text)
    if report_path:
        print(f"run report written to {report_path}")


if __name__ == "__main__":
    main()
