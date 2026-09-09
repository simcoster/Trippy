"""Embed parks.org.il #breadcrumbs as region claims.

No review row. `notes` is BREADCRUMB_NOTES so they survive clear-claims
and can be rebuilt by this scrape. The claim splitter drops these
sentences; the embedder and claim judge keep them
(experiments.md 2026-09-09 §1).

  uv run python -m source.scraper.info_site.breadcrumbs --site 5
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from db.connect import connect, database_url
from source.scraper.amenity_enrichment.llm import (
    ClaimsEmbeddingLLMClient,
    LlmUsage,
    make_nebius_openai_client,
    record_scrape_cost,
)
from source.scraper.cli import add_site_argument, site_ids
from source.scraper.info_site.parse import parse_breadcrumb_regions
from source.scraper.rules_ingest.fetch import fetch_page_html
from source.scraper.rules_ingest.ingest import (
    DEFAULT_LIMIT,
    DEFAULT_PAUSE_SECONDS,
    fetch_campsites,
    load_config,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

BREADCRUMB_NOTES = "no review, region by breadcrumbs"

DELETE_BREADCRUMB_CLAIMS_SQL = """
DELETE FROM claims
WHERE campsite_id = %(campsite_id)s
  AND notes = %(notes)s
"""

INSERT_BREADCRUMB_CLAIM_SQL = """
INSERT INTO claims (
    review_id, campsite_id, claim, evidence_span,
    is_positive, confidence, notes, embedding
) VALUES (
    NULL, %(campsite_id)s, %(claim)s, %(evidence_span)s,
    TRUE, 1.0, %(notes)s, %(embedding)s
)
"""


def snapshot_breadcrumb_claims(
    conn,
    *,
    campsite_id: int,
    regions: list[dict],
    embedder: ClaimsEmbeddingLLMClient,
    usage: LlmUsage | None = None,
) -> int:
    """Replace this site's breadcrumb claims. Embed, then delete+insert."""
    rows = [row for row in regions if row.get("slug")]
    slugs = [str(row["slug"]) for row in rows]
    vectors = embedder.embed(slugs, usage=usage) if slugs else []
    with conn.cursor() as cur:
        cur.execute(
            DELETE_BREADCRUMB_CLAIMS_SQL,
            {"campsite_id": campsite_id, "notes": BREADCRUMB_NOTES},
        )
        for region, vector in zip(rows, vectors, strict=True):
            cur.execute(
                INSERT_BREADCRUMB_CLAIM_SQL,
                {
                    "campsite_id": campsite_id,
                    "claim": region["slug"],
                    "evidence_span": region.get("label") or None,
                    "notes": BREADCRUMB_NOTES,
                    "embedding": vector,
                },
            )
    return len(vectors)


def run(
    config: dict,
    *,
    limit: int,
    sites: list[int] | None = None,
    usage: LlmUsage | None = None,
) -> int:
    campsites = fetch_campsites(config, limit=limit, sites=sites)
    if not campsites:
        print("No campsites found")
        return 0
    pause_s = float(
        config.get("info_site", {}).get("request_pause_seconds", DEFAULT_PAUSE_SECONDS)
    )
    usage = usage if usage is not None else LlmUsage()
    embedder = ClaimsEmbeddingLLMClient(make_nebius_openai_client())
    total = 0
    print(f"Scraping breadcrumb regions for {len(campsites)} campsite(s)")
    with connect(database_url(config)) as conn:
        register_vector(conn)
        for site in campsites:
            print("=" * 60)
            print(f"{site['id']}. {site['name']}")
            print(f"   {site['url']}")
            try:
                html = fetch_page_html(site["url"])
            except httpx.HTTPError as exc:
                print(f"    HTTP error: {exc}")
                continue
            regions = parse_breadcrumb_regions(html)
            saved = snapshot_breadcrumb_claims(
                conn,
                campsite_id=site["id"],
                regions=regions,
                embedder=embedder,
                usage=usage,
            )
            conn.commit()
            total += saved
            slugs = ", ".join(r["slug"] for r in regions) or "(none)"
            print(f"    {saved} region claim(s): {slugs}")
            if pause_s > 0:
                time.sleep(pause_s)
    print("-" * 60)
    print(f"Done. Stored {total} breadcrumb region claim(s).")
    if usage.chat_calls or usage.embed_calls:
        print(usage.summary(prefix="Breadcrumbs total: "))
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parks.org.il breadcrumb region claims")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="How many campsites to process (default: info_site.limit_campsites)",
    )
    add_site_argument(parser)
    args = parser.parse_args(argv)
    config = load_config()
    limit = args.limit
    if limit is None:
        limit = int(config.get("info_site", {}).get("limit_campsites", DEFAULT_LIMIT))
    usage = LlmUsage()
    run(config, limit=limit, sites=site_ids(args.site), usage=usage)
    written = record_scrape_cost("scrape-breadcrumbs", usage)
    if written:
        print(f"cost report appended to {written}")


if __name__ == "__main__":
    main()
