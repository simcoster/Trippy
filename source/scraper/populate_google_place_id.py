"""Resolve each campsite name via legacy Places Text Search; store google_place_id.

Phase one: take the first hit. Overnight campgrounds inside a nature reserve
often share a park listing, or have a separate pin — Text Search on
campsites.name usually returns exactly one of those, not both.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import httpx
import psycopg
from dotenv import load_dotenv

from source.scraper.tls import ssl_context

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

TEXTSEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
DEFAULT_PAUSE_SECONDS = 0.25

SELECT_CAMPSITES_SQL = """
SELECT id, name, google_place_id
FROM campsites
WHERE (%(campsite_id)s::bigint IS NULL OR id = %(campsite_id)s)
  AND (%(force)s::boolean OR google_place_id IS NULL)
ORDER BY id
LIMIT %(limit)s
"""

UPDATE_PLACE_ID_SQL = """
UPDATE campsites
SET google_place_id = %(google_place_id)s
WHERE id = %(id)s
RETURNING id, name, google_place_id
"""


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        url = "postgresql://trippy:trippy@localhost:5432/trippy"
    return url.replace("@db:", "@localhost:")


def google_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GOOGLE_API_KEY is required")
    return key


def log(msg: str) -> None:
    print(msg, flush=True)


def first_place_from_textsearch(body: dict) -> dict | None:
    """Phase one: first result only. Empty / non-OK with no results → None."""
    results = body.get("results")
    if not isinstance(results, list) or not results:
        return None
    hit = results[0]
    if not isinstance(hit, dict):
        return None
    place_id = hit.get("place_id")
    if not place_id:
        return None
    return {
        "place_id": str(place_id),
        "name": hit.get("name"),
        "rating": hit.get("rating"),
        "user_ratings_total": hit.get("user_ratings_total"),
        "types": hit.get("types") or [],
        "formatted_address": hit.get("formatted_address"),
        "n_hits": len(results),
    }


def textsearch(
    client: httpx.Client,
    query: str,
    api_key: str,
) -> dict:
    response = client.get(
        TEXTSEARCH_URL,
        params={
            "query": query,
            "language": "he",
            "region": "il",
            "key": api_key,
        },
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("Text Search response is not a JSON object")
    return body


def fetch_campsites(
    conn,
    *,
    campsite_id: int | None = None,
    force: bool = False,
    limit: int | None = None,
) -> list[tuple[int, str, str | None]]:
    cap = limit if limit is not None else 10_000
    with conn.cursor() as cur:
        cur.execute(
            SELECT_CAMPSITES_SQL,
            {
                "campsite_id": campsite_id,
                "force": force,
                "limit": cap,
            },
        )
        return [(int(row[0]), str(row[1]), row[2]) for row in cur.fetchall()]


def store_google_place_id(
    cur,
    *,
    campsite_id: int,
    place_id: str,
) -> tuple[int, str, str]:
    cur.execute(
        UPDATE_PLACE_ID_SQL,
        {"id": campsite_id, "google_place_id": place_id},
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"campsite id={campsite_id} not found")
    return int(row[0]), str(row[1]), str(row[2])


def populate_google_place_ids(
    *,
    conn=None,
    client: httpx.Client | None = None,
    api_key: str | None = None,
    campsite_id: int | None = None,
    force: bool = False,
    limit: int | None = None,
    pause_seconds: float = DEFAULT_PAUSE_SECONDS,
) -> dict[str, Any]:
    """Text Search each pending campsite name; write the first hit's place_id."""
    own_conn = conn is None
    own_client = client is None
    if own_conn:
        conn = psycopg.connect(database_url())
    if own_client:
        client = httpx.Client(verify=ssl_context(), timeout=30.0)
    key = api_key if api_key is not None else google_api_key()

    updated: list[dict] = []
    skipped: list[dict] = []
    try:
        sites = fetch_campsites(
            conn, campsite_id=campsite_id, force=force, limit=limit
        )
        if not sites:
            log("No campsites to resolve")
            return {"updated": [], "skipped": []}

        log(f"Resolving {len(sites)} campsite(s)")
        with conn.cursor() as cur:
            for i, (site_id, name, existing) in enumerate(sites):
                if i and pause_seconds:
                    time.sleep(pause_seconds)
                body = textsearch(client, name, key)
                status = body.get("status")
                hit = first_place_from_textsearch(body)
                if hit is None:
                    skipped.append(
                        {
                            "id": site_id,
                            "name": name,
                            "status": status,
                            "existing": existing,
                        }
                    )
                    log(f"  {site_id}. {name}  status={status}  no place_id")
                    continue
                store_google_place_id(
                    cur, campsite_id=site_id, place_id=hit["place_id"]
                )
                row = {
                    "id": site_id,
                    "name": name,
                    "google_place_id": hit["place_id"],
                    "google_name": hit.get("name"),
                    "n_hits": hit["n_hits"],
                    "status": status,
                }
                updated.append(row)
                extra = f"  ({hit['n_hits']} hits)" if hit["n_hits"] > 1 else ""
                log(
                    f"  {site_id}. {name}  →  {hit.get('name')}  "
                    f"{hit['place_id']}{extra}"
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if own_client:
            client.close()
        if own_conn:
            conn.close()

    log(f"Updated {len(updated)}; skipped {len(skipped)}")
    return {"updated": updated, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy Places Text Search on campsites.name; store google_place_id "
            "(first hit)."
        )
    )
    parser.add_argument("--campsite-id", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite google_place_id even when already set",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    populate_google_place_ids(
        campsite_id=args.campsite_id,
        force=args.force,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
