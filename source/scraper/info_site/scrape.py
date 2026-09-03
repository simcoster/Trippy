"""
Scrape published rate cards from parks.org.il camping info pages.

Creates info_website_names from classified lodging rows and snapshots
list_prices. Does not create accommodation_types or scrape newsflashes.

  uv run python source/scraper/info_site/scrape.py --prices
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
from pathlib import Path

import httpx
import psycopg
from amenity_enrichment.llm import LlmUsage
from dotenv import load_dotenv

_SCRAPER_DIR = Path(__file__).resolve().parents[1]
if str(_SCRAPER_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRAPER_DIR))

from info_site.classify import RateCardClassifier, classify_rows  # noqa: E402
from info_site.db import maybe_fill_booking_hotel_id, snapshot_list_prices  # noqa: E402
from info_site.parse import (  # noqa: E402
    parse_booking_hotel_id,
    parse_rate_table,
    parse_wp_post_id,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

CONFIG_PATH = _SCRAPER_DIR / "config.json"
LISTING_URL = (
    "https://www.parks.org.il/"
    "%D7%94%D7%96%D7%9E%D7%A0%D7%95%D7%AA-%D7%9C%D7%97%D7%A0%D7%99%D7%95%D7%A0%D7%99-%D7%9C%D7%99%D7%9C%D7%94/"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)


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
    limit = int(config.get("info_site", {}).get("limit_campsites", 2))
    sql = """
        SELECT id, name, url, booking_hotel_id
        FROM campsites
        -- Subcamps have no page of their own; their parent's rate card covers
        -- them, and scraping a NULL url would fail.
        WHERE url IS NOT NULL
        ORDER BY id
        LIMIT %s
    """
    with psycopg.connect(database_url(config)) as conn, conn.cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "name": row[1],
            "url": row[2],
            "booking_hotel_id": row[3],
        }
        for row in rows
    ]


def fetch_page_html(url: str, *, referer: str = LISTING_URL) -> str:
    with httpx.Client(
        timeout=45.0,
        verify=_ssl_context(),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Referer": referer},
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def scrape_prices_for_site(
    conn,
    site: dict,
    html: str,
    *,
    classifier: RateCardClassifier,
    usage: LlmUsage | None = None,
) -> int:
    raw_rows = parse_rate_table(html)
    classified = classify_rows(raw_rows, classifier=classifier, usage=usage)
    lodging = snapshot_list_prices(conn, site_id=site["id"], rows=classified)
    hotel_id = parse_booking_hotel_id(html)
    with conn.cursor() as cur:
        filled = maybe_fill_booking_hotel_id(
            cur, site_id=site["id"], booking_hotel_id=hotel_id
        )
    if filled:
        print(f"    filled booking_hotel_id={filled}")
    post_id = parse_wp_post_id(html)
    if post_id:
        print(f"    wp post id={post_id}")
    fees = sum(1 for row in classified if row.kind == "fee")
    print(f"    {len(raw_rows)} table rows, {len(lodging)} lodging stored, {fees} fees skipped")
    return len(lodging)


def run_prices(config: dict) -> int:
    campsites = fetch_campsites(config)
    if not campsites:
        print("No campsites found")
        return 0

    pause_s = float(config.get("info_site", {}).get("request_pause_seconds", 0.5))
    classifier = RateCardClassifier()
    usage = LlmUsage()
    total = 0

    print(f"Scraping list prices for {len(campsites)} campsite(s)")
    with psycopg.connect(database_url(config)) as conn:
        for site in campsites:
            print("=" * 60)
            print(f"{site['id']}. {site['name']}")
            print(f"   {site['url']}")
            try:
                html = fetch_page_html(site["url"])
            except httpx.HTTPError as exc:
                print(f"    HTTP error: {exc}")
                continue
            saved = scrape_prices_for_site(
                conn, site, html, classifier=classifier, usage=usage
            )
            conn.commit()
            total += saved
            if pause_s > 0:
                time.sleep(pause_s)

    print("-" * 60)
    print(f"Done. Stored {total} lodging list-price row(s).")
    if usage.chat_calls:
        print(usage.summary(prefix="Classify total: "))
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parks.org.il info-site scraper")
    parser.add_argument(
        "--prices",
        action="store_true",
        help="Scrape רגיל rate-card tables into list_prices",
    )
    args = parser.parse_args(argv)
    if not args.prices:
        parser.error("pass --prices (newsflashes are not wired yet)")
    run_prices(load_config())


if __name__ == "__main__":
    main()
