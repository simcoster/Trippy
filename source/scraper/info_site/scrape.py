"""
Scrape published rate cards from parks.org.il camping info pages.

Creates info_website_names from classified lodging rows and snapshots
list_prices. Does not create accommodation_types or scrape newsflashes.

  uv run python -m source.scraper.info_site.scrape --prices
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from db.connect import connect, database_url
from source.scraper.amenity_enrichment.llm import LlmUsage, record_scrape_cost
from source.scraper.cli import add_site_argument, site_ids
from source.scraper.info_site.classify import RateCardClassifier, classify_rows
from source.scraper.info_site.db import (
    UNCERTAIN_BELOW,
    maybe_fill_booking_hotel_id,
    snapshot_list_prices,
)
from source.scraper.info_site.match_listing import InfoWebsiteNameMatcher, MatchCall
from source.scraper.info_site.parse import (
    parse_booking_hotel_id,
    parse_rate_table,
    parse_wp_post_id,
)
from source.scraper.tls import ssl_context

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

_SCRAPER_DIR = Path(__file__).resolve().parents[1]
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


def fetch_campsites(config: dict, *, sites: list[int] | None = None) -> list[dict]:
    """The pages to scrape: the ones `sites` names, else the first `limit`.

    `--site` exists so `just scrape-info -- --site 5,14` means the same sites at
    every step of the pipeline; without it the rooms step would run on two and
    the prices step on twenty.
    """
    limit = int(config.get("info_site", {}).get("limit_campsites", 2))
    # Subcamps have no page of their own; their parent's rate card covers them,
    # and scraping a NULL url would fail.
    where = "WHERE url IS NOT NULL" + (" AND id = ANY(%(sites)s)" if sites else "")
    sql = f"""
        SELECT id, name, url, booking_hotel_id
        FROM campsites
        {where}
        ORDER BY id
        LIMIT %(limit)s
    """
    with connect(database_url(config)) as conn, conn.cursor() as cur:
        cur.execute(
            sql,
            {"sites": list(sites or ()), "limit": len(sites) if sites else limit},
        )
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
        verify=ssl_context(),
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
    matcher: InfoWebsiteNameMatcher | None = None,
    unmatched_sink: list[str] | None = None,
) -> int:
    raw_rows = parse_rate_table(html)
    classified = classify_rows(raw_rows, classifier=classifier, usage=usage)
    lodging = snapshot_list_prices(
        conn,
        site_id=site["id"],
        rows=classified,
        matcher=matcher,
        usage=usage,
        unmatched_sink=unmatched_sink,
    )
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


def match_verdict(call: MatchCall) -> str:
    """"forced", "collision", "split", "uncertain", or "" -- the same reading
    `snapshot_list_prices` makes of the answer, taken from the record rather
    than passed alongside it.

    A refusal is the model ignoring an instruction the prompt states plainly, so
    the price is forced onto a candidate; a low number is the model doing as it
    was told and saying the match is poor. A split is the rescue pass finding
    that one rate really does price several products -- confident or not, that
    is worth seeing, because it writes more rows than the rate card has lines.
    """
    if call.kind == "collision":
        # Always shown: the clash was established in code, so this is the one
        # answer with no confidence of its own to hide behind.
        return "collision" if call.picked is not None else "collision unresolved"
    if call.picked is None:
        return "forced"
    if len(call.picked_names) > 1:
        return "split"
    if call.confidence is not None and call.confidence < UNCERTAIN_BELOW:
        return "uncertain"
    return ""


def print_flagged_prompts(matcher: InfoWebsiteNameMatcher) -> None:
    """Every uncertain or forced match, with the exact prompt that produced it.

    A wrong pick is either the prompt's fault or the model's, and the summary
    lines above cannot tell you which: they show the answer, not the question.
    Only the user message is printed -- the system prompt is byte-identical on
    every call and lives in `match_listing.SYSTEM_PROMPT`.
    """
    flagged = [(call, match_verdict(call)) for call in matcher.calls]
    flagged = [(call, verdict) for call, verdict in flagged if verdict]
    if not flagged:
        return
    print()
    print("=" * 60)
    print(f"PROMPTS FOR {len(flagged)} FLAGGED MATCH(ES)")
    print("=" * 60)
    for i, (call, verdict) in enumerate(flagged, start=1):
        confidence = "none" if call.confidence is None else f"{call.confidence:.2f}"
        print("-" * 60)
        print(f"{i}. {verdict.upper()} (confidence {confidence}) -- {call.site}")
        if call.kind == "rescue" and len(call.picked_names) > 1:
            print(f"   priced against {len(call.picked_names)} products")
        print()
        print("[user]")
        print(call.user)
        print()
        print("[reply]")
        print(call.reply)
    print("-" * 60)


def run_prices(
    config: dict, *, usage: LlmUsage | None = None, sites: list[int] | None = None
) -> int:
    """Scrape rate cards for the configured campsites. Returns rows stored.

    `usage` collects every LLM call so the caller can report the run's cost.
    """
    campsites = fetch_campsites(config, sites=sites)
    if not campsites:
        print("No campsites found")
        return 0

    pause_s = float(config.get("info_site", {}).get("request_pause_seconds", 0.5))
    classifier = RateCardClassifier()
    usage = usage if usage is not None else LlmUsage()
    matcher = InfoWebsiteNameMatcher()
    unmatched: list[str] = []
    total = 0

    print(f"Scraping list prices for {len(campsites)} campsite(s)")
    with connect(database_url(config)) as conn:
        for site in campsites:
            print("=" * 60)
            print(f"{site['id']}. {site['name']}")
            print(f"   {site['url']}")
            try:
                html = fetch_page_html(site["url"])
            except httpx.HTTPError as exc:
                print(f"    HTTP error: {exc}")
                continue
            calls_before = len(matcher.calls)
            saved = scrape_prices_for_site(
                conn,
                site,
                html,
                classifier=classifier,
                usage=usage,
                matcher=matcher,
                unmatched_sink=unmatched,
            )
            for call in matcher.calls[calls_before:]:
                call.site = site["name"]
            conn.commit()
            total += saved
            if pause_s > 0:
                time.sleep(pause_s)

    print("-" * 60)
    print(f"Done. Stored {total} lodging list-price row(s).")
    if unmatched:
        print(f"{len(unmatched)} rate-card label(s) matched no lodging product:")
        for name in sorted(set(unmatched)):
            print(f"  {name}")
    print_flagged_prompts(matcher)
    if usage.chat_calls:
        print(usage.summary(prefix="Classify total: "))
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parks.org.il info-site scraper")
    add_site_argument(parser)
    parser.add_argument(
        "--prices",
        action="store_true",
        help="Scrape רגיל rate-card tables into list_prices",
    )
    args = parser.parse_args(argv)
    if not args.prices:
        parser.error("pass --prices (newsflashes are not wired yet)")
    usage = LlmUsage()
    run_prices(load_config(), usage=usage, sites=site_ids(args.site))
    written = record_scrape_cost("scrape-prices", usage)
    if written:
        print(f"cost report appended to {written}")


if __name__ == "__main__":
    main()
