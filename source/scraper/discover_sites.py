"""
Crawler to discover campsites from parks.org.il
Extracts campsite elements and upserts them into Postgres.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from db.connect import connect
from source.scraper.amenity_enrichment.llm import (
    LlmUsage,
    _parse_json_payload,
    instruct_chat_model,
    make_nebius_openai_client,
    record_scrape_cost,
)
from source.scraper.tls import ssl_context

# Windows consoles often default to cp1252 and choke on Hebrew titles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

SCRAPER_DIR = Path(__file__).resolve().parent
LISTING_URL = (
    "https://www.parks.org.il/"
    "%D7%94%D7%96%D7%9E%D7%A0%D7%95%D7%AA-%D7%9C%D7%97%D7%A0%D7%99%D7%95%D7%A0%D7%99-%D7%9C%D7%99%D7%9C%D7%94/"
)
ENGLISH_LISTING_URL = "https://en.parks.org.il/camping/"
OUTPUT_FILE = SCRAPER_DIR / "campsites.json"

_LISTING_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

UPSERT_SQL = """
INSERT INTO campsites (name, url)
VALUES (%(name)s, %(url)s)
ON CONFLICT (url) DO UPDATE
SET name = EXCLUDED.name
RETURNING id, name, url;
"""


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        # Local default when running the crawler on the host against Compose db
        url = "postgresql://trippy:trippy@localhost:5432/trippy"
    # Docker service hostname is unreachable from the host
    return url.replace("@db:", "@localhost:")


def _listing_soup(url: str) -> BeautifulSoup:
    with httpx.Client(
        timeout=30.0,
        verify=ssl_context(),
        follow_redirects=True,
    ) as client:
        response = client.get(url, headers=_LISTING_HEADERS)
        response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def crawl_campsites(url: str = LISTING_URL) -> list[dict[str, str]]:
    """Crawl the Hebrew listing page; return [{name, url}, ...]."""
    try:
        soup = _listing_soup(url)
        campsite_elements = soup.find_all(
            "div", class_=lambda c: c and "team_repeater_wrapper" in c
        )
        print(f"Found {len(campsite_elements)} campsite elements")

        campsites: list[dict[str, str]] = []
        for campsite in campsite_elements:
            href = campsite.select_one("a")["href"]
            title = campsite.select_one("h2").get_text(strip=True)

            if href and not href.startswith("http"):
                href = urljoin(url, href)

            if href and title:
                campsites.append({"name": title, "url": href})
                print(f"Found: {title[:50]}... -> {href[:80]}...")
        return campsites

    except httpx.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return []


def english_listing_names_from_html(html: str) -> list[str]:
    """h2 titles inside the English listing cards. Order preserved, deduped."""
    soup = BeautifulSoup(html, "html.parser")
    names: list[str] = []
    seen: set[str] = set()
    for heading in soup.select("div.article_content h2"):
        title = heading.get_text(strip=True)
        if not title or title in seen:
            continue
        seen.add(title)
        names.append(title)
    return names


def crawl_english_listing_names(url: str = ENGLISH_LISTING_URL) -> list[str]:
    """English listing titles only (en.parks.org.il/camping/). Deduped."""
    try:
        soup = _listing_soup(url)
        names = english_listing_names_from_html(soup.decode())
        print(f"Found {len(names)} English listing name(s)")
        for title in names:
            print(f"  {title}")
        return names
    except httpx.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return []


def upsert_campsites(campsites: list[dict[str, str]]) -> list[dict]:
    """Insert or update campsites by URL; return rows with generated ids."""
    if not campsites:
        return []

    db_url = _database_url()
    saved: list[dict] = []
    with connect(db_url) as conn:
        with conn.cursor() as cur:
            for site in campsites:
                cur.execute(UPSERT_SQL, site)
                row = cur.fetchone()
                if row:
                    saved.append({"id": row[0], "name": row[1], "url": row[2]})
        conn.commit()
    return saved


UPDATE_ENGLISH_SQL = """
UPDATE campsites
SET english_name = %(english_name)s
WHERE id = %(id)s
"""

ENGLISH_NAME_SYSTEM = """
You match Israeli INPA overnight campsites.

The user JSON has:
- hebrew: [{id, name}, ...] Hebrew titles from the database (some rows are
  subcamps of one park).
- english_listing: [...] titles copied from en.parks.org.il/camping/.
  Closed list. The base English name must be one of these strings exactly.

Return JSON only: an object mapping each id as a string to an English name
or null. Use null when no listing title is the same park.

If the Hebrew name is a north/south subcamp, take that park's listing
title and append " North" or " South". Do not invent any other wording.
""".strip()

UPSERT_SUBCAMP_SQL = """
INSERT INTO campsites (name, parent_id, subcamp)
VALUES (%(name)s, %(parent_id)s, %(subcamp)s::jsonb)
ON CONFLICT (parent_id, name) WHERE parent_id IS NOT NULL DO UPDATE
SET subcamp = EXCLUDED.subcamp
RETURNING id, name;
"""


def load_subcamp_config(path: Path = SCRAPER_DIR / "config.json") -> dict:
    """The `subcamps` block: which sites are split, keyed by page URL.

    Configuration rather than detection, and deliberately so — see
    docs/design.md. Being in the repo it also survives repopulating an empty
    database, which is the case that matters when this moves to the cloud.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("subcamps") or {}
    except FileNotFoundError:
        return {}


def upsert_subcamps(saved: list[dict]) -> int:
    """Give every configured split site one child row per subcamp.

    A child carries no `url` or `booking_hotel_id`: both subcamps share one
    page and one booking id, and leaving those NULL is what makes every scraper
    filtering `WHERE url IS NOT NULL` skip children without knowing they exist.
    Reviews, claims and prices stay on the parent for the same reason — a guest
    review says "Akhziv", not "the northern one".
    """
    config = load_subcamp_config()
    if not config:
        return 0
    by_url = {site["url"]: site for site in saved}
    written = 0
    with connect(_database_url()) as conn:
        with conn.cursor() as cur:
            for url, areas in config.items():
                parent = by_url.get(url)
                if parent is None:
                    print(f"    subcamp config matches no campsite: {url[:70]}")
                    continue
                for area in areas:
                    cur.execute(
                        UPSERT_SUBCAMP_SQL,
                        {
                            "name": f"{parent['name']} – {area['heading']}",
                            "parent_id": parent["id"],
                            "subcamp": json.dumps(area, ensure_ascii=False),
                        },
                    )
                    row = cur.fetchone()
                    if row:
                        written += 1
                        print(f"    subcamp {row[0]}: {row[1]}")
        conn.commit()
    return written


def _english_by_id(parsed: dict[str, Any]) -> dict[int, str]:
    raw: Any = parsed
    if len(parsed) == 1:
        only = next(iter(parsed.values()))
        if isinstance(only, dict):
            raw = only
    if not isinstance(raw, dict):
        return {}
    out: dict[int, str] = {}
    for key, value in raw.items():
        try:
            campsite_id = int(key)
        except (TypeError, ValueError):
            continue
        text = str(value or "").strip()
        if not text or text.casefold() == "null":
            continue
        out[campsite_id] = text
    return out


def fill_english_names(
    english_listing: list[str],
    *,
    client: Any | None = None,
    usage: LlmUsage | None = None,
) -> int:
    """One 235B call: match Hebrew `campsites.name` to the English listing."""
    if not english_listing:
        print("english_name: no English listing names; skip")
        return 0
    db_url = _database_url()
    with connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name FROM campsites ORDER BY id")
            rows = list(cur.fetchall())
    if not rows:
        return 0
    payload = {
        "hebrew": [{"id": int(row[0]), "name": row[1]} for row in rows],
        "english_listing": english_listing,
    }
    model = instruct_chat_model()
    api = client or make_nebius_openai_client()
    response = api.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": ENGLISH_NAME_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    if usage is not None:
        usage.add_chat(response.usage, role="campsite_english_name", model=model)
    raw = (response.choices[0].message.content or "").strip()
    try:
        parsed = _parse_json_payload(raw)
    except (json.JSONDecodeError, ValueError):
        print(f"english_name: unparseable model output: {raw[:200]}")
        return 0
    mapping = _english_by_id(parsed)
    written = 0
    with connect(db_url) as conn:
        with conn.cursor() as cur:
            for campsite_id, english_name in mapping.items():
                cur.execute(
                    UPDATE_ENGLISH_SQL,
                    {"id": campsite_id, "english_name": english_name},
                )
                written += cur.rowcount
        conn.commit()
    return written


def main():
    print(f"Crawling: {LISTING_URL}")
    print("-" * 80)

    campsites = crawl_campsites()
    print("-" * 80)
    print(f"\nTotal campsites found: {len(campsites)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(campsites, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")

    if not campsites:
        return

    saved = upsert_campsites(campsites)
    print(f"Upserted {len(saved)} campsites into Postgres")

    subcamps = upsert_subcamps(saved)
    if subcamps:
        print(f"Upserted {subcamps} subcamp row(s) from config")

    usage = LlmUsage()
    print(f"Crawling English names: {ENGLISH_LISTING_URL}")
    english_listing = crawl_english_listing_names()
    named = fill_english_names(english_listing, usage=usage)
    print(f"Set english_name on {named} campsite(s)")
    cost_path = record_scrape_cost("scrape-sites", usage)
    if cost_path:
        print(f"cost report appended to {cost_path}")

    print("\nFirst 5 DB rows:")
    for site in saved[:5]:
        print(f"{site['id']}. {site['name']}")
        print(f"   {site['url']}\n")


if __name__ == "__main__":
    main()
