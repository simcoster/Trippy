"""site_price_functions upsert returns the row's scraped_at / updated_at.

One row per campsite. A second scrape with the same hash only bumps
scraped_at. Writes stay in `experiments`.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.info_site.db import store_price_function
from source.scraper.info_site.price_report import PriceFunctionRun, render_run_report
from source.test.test_cases.experiments_schema import SITE_ID, clone_tables


@pytest.fixture
def conn(experiments_site):
    connection, _site_id = experiments_site
    with connection.cursor() as cur:
        clone_tables(cur, ("campsites", "site_price_functions"))
        cur.execute(
            "INSERT INTO campsites (id, name, url) VALUES (%s, %s, %s)",
            (SITE_ID, "test campsite", "https://example.invalid/test"),
        )
    return connection


def test_insert_returns_both_timestamps(conn):
    stored = store_price_function(
        conn, site_id=SITE_ID, source="def quote():\n    return 1\n", digest="aaa"
    )
    assert stored.status == "inserted"
    assert stored.scraped_at is not None
    assert stored.updated_at == stored.scraped_at


def test_new_hash_updates_both_timestamps(conn):
    store_price_function(
        conn, site_id=SITE_ID, source="def quote():\n    return 1\n", digest="aaa"
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE site_price_functions
            SET scraped_at = now() - interval '1 hour',
                updated_at = now() - interval '1 hour'
            WHERE site_id = %s
            RETURNING scraped_at, updated_at
            """,
            (SITE_ID,),
        )
        old_scraped, old_updated = cur.fetchone()
    second = store_price_function(
        conn, site_id=SITE_ID, source="def quote():\n    return 2\n", digest="bbb"
    )
    assert second.status == "updated"
    assert second.scraped_at > old_scraped
    assert second.updated_at > old_updated


def test_same_hash_bumps_scraped_at_only(conn):
    store_price_function(
        conn, site_id=SITE_ID, source="def quote():\n    return 1\n", digest="aaa"
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE site_price_functions
            SET scraped_at = now() - interval '1 hour',
                updated_at = now() - interval '1 hour'
            WHERE site_id = %s
            """,
            (SITE_ID,),
        )
    stored = store_price_function(
        conn, site_id=SITE_ID, source="def quote():\n    return 1\n", digest="aaa"
    )
    assert stored.status == "unchanged"
    assert stored.scraped_at > stored.updated_at


def test_report_prints_store_times():
    when = datetime(2026, 9, 17, 12, 18, 28)
    text = render_run_report(
        [
            PriceFunctionRun(
                site_id=2,
                site_name="אכזיב",
                outcome="stored",
                store_status="updated",
                digest="abcdef123456",
                scraped_at=when,
                updated_at=when,
            )
        ],
        LlmUsage(),
        started_at=when,
        seconds=1,
    )
    assert "scraped_at: 2026-09-17T12:18:28" in text
    assert "updated_at: 2026-09-17T12:18:28" in text
