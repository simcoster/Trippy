"""Failing tomorrow-target: first campsite rooms must carry amenity ids with embeddings."""

from __future__ import annotations

import os

import psycopg
import pytest
from dotenv import load_dotenv

load_dotenv()


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    assert url, "DATABASE_URL is required"
    return url.replace("@db:", "@localhost:")


@pytest.fixture
def conn():
    with psycopg.connect(_db_url()) as connection:
        yield connection


def test_first_site_accommodation_has_amenities_with_embeddings(conn):
    """First campsite has ≥1 accommodation type whose amenity ids exist and are embedded."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, name FROM campsites ORDER BY id LIMIT 1")
        site = cur.fetchone()
        assert site is not None, "expected at least one campsite"

        site_id, site_name = site
        cur.execute(
            """
            SELECT id, name, amenities
            FROM accommodation_types
            WHERE hotel_id = %s
            ORDER BY id
            """,
            (site_id,),
        )
        types = cur.fetchall()
        assert types, (
            f"campsite {site_id} ({site_name!r}) has no accommodation_types"
        )

        accom_id, accom_name, amenity_ids = types[0]
        assert amenity_ids, (
            f"accommodation_type {accom_id} ({accom_name!r}) has empty amenities"
        )
        assert isinstance(amenity_ids, list), (
            f"amenities should be a JSON list of ids, got {type(amenity_ids).__name__}"
        )
        assert all(isinstance(a, int) for a in amenity_ids), (
            f"amenities must be amenity ids (ints), got {amenity_ids!r}"
        )

        cur.execute(
            """
            SELECT id, name, embedding IS NOT NULL AS has_embedding
            FROM amenities
            WHERE id = ANY(%s)
            ORDER BY id
            """,
            (amenity_ids,),
        )
        rows = cur.fetchall()
        found_ids = {row[0] for row in rows}
        missing = sorted(set(amenity_ids) - found_ids)
        assert not missing, f"amenity ids not in amenities table: {missing}"

        without_embedding = [
            {"id": row[0], "name": row[1]} for row in rows if not row[2]
        ]
        assert not without_embedding, (
            f"amenities missing embeddings: {without_embedding}"
        )
