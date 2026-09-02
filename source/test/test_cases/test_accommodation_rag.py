"""RAG search over accommodation amenity embeddings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

FIXTURES_DIR = Path(__file__).resolve().parent / "test_cases"
BARBECUE_QUERY_CACHE = FIXTURES_DIR / "barbecue_query_embedding.json"


@dataclass
class AccommodationMatch:
    id: int
    name: str
    matched_amenity: str
    distance: float


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    assert url, "DATABASE_URL is required"
    return url.replace("@db:", "@localhost:")


@lru_cache
def load_barbecue_query_embedding() -> tuple[str, list[float]]:
    """Load cached query text + embedding (see test_cases/barbecue_query_embedding.json)."""
    data = json.loads(BARBECUE_QUERY_CACHE.read_text(encoding="utf-8"))
    query = data["query"]
    embedding = data["embedding"]
    assert isinstance(query, str) and query.strip(), "cache must include query text"
    assert isinstance(embedding, list) and embedding, "cache must include embedding"
    assert len(embedding) == data.get("dimensions", len(embedding))
    return query, embedding


def search_accommodations_by_amenity(
    conn,
    embedding: list[float],
    *,
    limit: int = 5,
) -> list[AccommodationMatch]:
    """Rank accommodation types by closest embedded amenity to the query vector."""
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT at.id,
                   at.name,
                   MIN(a.embedding <#> %s::vector) AS distance,
                   (array_agg(a.name ORDER BY a.embedding <#> %s::vector))[1]
                       AS matched_amenity
            FROM accommodation_types at
            CROSS JOIN LATERAL jsonb_array_elements(at.amenities) AS elem(val)
            JOIN subject_vectors a ON a.id = (elem.val)::int
            WHERE a.embedding IS NOT NULL
              AND at.amenities IS NOT NULL
            GROUP BY at.id, at.name
            ORDER BY distance
            LIMIT %s
            """,
            (embedding, embedding, limit),
        )
        rows = cur.fetchall()
    return [
        AccommodationMatch(
            id=int(row[0]),
            name=row[1],
            distance=float(row[2]),
            matched_amenity=row[3],
        )
        for row in rows
    ]


@pytest.fixture
def conn():
    with psycopg.connect(_db_url()) as connection:
        yield connection


def test_barbecue_query_returns_trailer_parking_spot(conn):
    """Semantic amenity search: 'barbecue' → trailer spot (id=1) via fire pit."""
    query, embedding = load_barbecue_query_embedding()
    assert query == "barbecue"

    results = search_accommodations_by_amenity(conn, embedding, limit=5)
    assert results, "expected at least one accommodation match"

    top = results[0]
    assert top.id == 1, (
        f"expected trailer parking spot id=1, got id={top.id} ({top.name!r})"
    )
    assert top.matched_amenity == "barbecue_pit", (
        f"expected fire-pit amenity match, got {top.matched_amenity!r}"
    )
