"""
Crawler to discover campsites from parks.org.il
Extracts campsite elements and upserts them into Postgres.
"""

import json
import os
import ssl
import sys
from pathlib import Path
from urllib.parse import urljoin

import httpx
import psycopg
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Windows consoles often default to cp1252 and choke on Hebrew titles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

SCRAPER_DIR = Path(__file__).resolve().parent
LISTING_URL = (
    "https://www.parks.org.il/"
    "%D7%94%D7%96%D7%9E%D7%A0%D7%95%D7%AA-%D7%9C%D7%97%D7%A0%D7%99%D7%95%D7%A0%D7%99-%D7%9C%D7%99%D7%9C%D7%94/"
)
OUTPUT_FILE = SCRAPER_DIR / "campsites.json"

UPSERT_SQL = """
INSERT INTO campsites (name, url)
VALUES (%(name)s, %(url)s)
ON CONFLICT (url) DO UPDATE
SET name = EXCLUDED.name
RETURNING id, name, url;
"""


def _ssl_context() -> ssl.SSLContext:
    """
    TLS context that verifies via the OS trust store (Windows certs), not
    certifi alone — MITM appliances (Norton, Zscaler, etc.) inject a local
    root that browsers trust but Mozilla's bundle does not.

    Also clears Python 3.13+'s VERIFY_X509_STRICT flag; those extra RFC
    checks often fail on MITM-rewritten chains that browsers still accept.
    """
    ctx = ssl.create_default_context()
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        # Local default when running the crawler on the host against Compose db
        url = "postgresql://trippy:trippy@localhost:5432/trippy"
    # Docker service hostname is unreachable from the host
    return url.replace("@db:", "@localhost:")


def crawl_campsites(url: str = LISTING_URL) -> list[dict[str, str]]:
    """Crawl the listing page; return [{name, url}, ...]."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    try:
        with httpx.Client(
            timeout=30.0,
            verify=_ssl_context(),
            follow_redirects=True,
        ) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
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


def upsert_campsites(campsites: list[dict[str, str]]) -> list[dict]:
    """Insert or update campsites by URL; return rows with generated ids."""
    if not campsites:
        return []

    db_url = _database_url()
    saved: list[dict] = []
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            for site in campsites:
                cur.execute(UPSERT_SQL, site)
                row = cur.fetchone()
                if row:
                    saved.append({"id": row[0], "name": row[1], "url": row[2]})
        conn.commit()
    return saved


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

    print("\nFirst 5 DB rows:")
    for site in saved[:5]:
        print(f"{site['id']}. {site['name']}")
        print(f"   {site['url']}\n")


if __name__ == "__main__":
    main()
