"""Skip bumps scraped_at only; a vacancy rewrite bumps updated_at too."""

from __future__ import annotations

from source.scraper import populate_availability as pa


def test_skip_touch_sql_does_not_set_updated_at():
    assert "scraped_at = now()" in pa.TOUCH_AVAILABILITY_SQL
    assert "updated_at" not in pa.TOUCH_AVAILABILITY_SQL
    assert "scraped_at = now()" in pa.TOUCH_PAGE_HASH_SQL
    assert "updated_at" not in pa.TOUCH_PAGE_HASH_SQL


def test_rewrite_sql_sets_both_timestamps():
    assert "scraped_at = now()" in pa.UPSERT_AVAILABILITY_SQL
    assert "updated_at = now()" in pa.UPSERT_AVAILABILITY_SQL
    assert "scraped_at = now()" in pa.UPSERT_PAGE_HASH_SQL
    assert "updated_at = now()" in pa.UPSERT_PAGE_HASH_SQL
