"""RAG search over accommodation amenity embeddings."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from source.agent.search import search_stated_amenities

load_dotenv()

# This file already lives in test_cases/, so the fixtures sit beside it.
FIXTURES_DIR = Path(__file__).resolve().parent
BARBECUE_QUERY_CACHE = FIXTURES_DIR / "barbecue_query_embedding.json"


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    assert url, "DATABASE_URL is required"
    return url.replace("@db:", "@localhost:")


@lru_cache
def load_barbecue_query_embedding() -> tuple[str, str]:
    """Cached query text + embedding, so the test costs no LLM call.

    Returned as a pgvector literal string, which is what
    `search_stated_amenities(embedding=...)` takes.
    """
    data = json.loads(BARBECUE_QUERY_CACHE.read_text(encoding="utf-8"))
    query = data["query"]
    embedding = data["embedding"]
    assert isinstance(query, str) and query.strip(), "cache must include query text"
    assert isinstance(embedding, list) and embedding, "cache must include embedding"
    assert len(embedding) == data.get("dimensions", len(embedding))
    return query, "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def trailer_parking_type_id() -> int:
    """The unit type that offers trailer parking, found by its own amenity.

    Not a hardcoded id: this test used to assert `id == 1`, which stopped being
    the trailer spot (id 1 is the tent pitch, id 2 the caravan bay) and turned a
    data change into a search failure.
    """
    with psycopg.connect(_db_url()) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT cr.accommodation_type_id
            FROM campsite_rules cr
            JOIN subject_vectors sv ON sv.id = cr.subject_id
            WHERE sv.name = 'trailer_parking'
              AND cr.accommodation_type_id IS NOT NULL
            ORDER BY cr.accommodation_type_id
            LIMIT 1
            """
        )
        row = cur.fetchone()
    assert row is not None, "no accommodation type offers trailer_parking"
    return int(row[0])


def test_barbecue_query_returns_trailer_parking_spot():
    """Semantic amenity search: 'barbecue' → the trailer spot, via its grill.

    Calls the shipped `search_stated_amenities` rather than a copy of its SQL.
    The copy this test used to carry still read `accommodation_types.amenities`
    and broke silently when migration 027 dropped it — the point of a search
    test is the query that actually ships.

    The amenity is matched by shape, not by name: the subject was
    `barbecue_pit` when this was written and is `personal_grill_station` now.
    What the test is really asserting is that an English query reaches a Hebrew
    tooltip's grill through the embedding, whatever that subject ends up called.
    """
    query, embedding = load_barbecue_query_embedding()
    assert query == "barbecue"

    results = search_stated_amenities(query, limit=5, embedding=embedding)
    assert results, "expected at least one accommodation match"
    assert "error" not in results[0], results[0]

    top = results[0]
    assert top["accommodation_type_id"] == trailer_parking_type_id(), (
        f"expected the trailer parking spot, got "
        f"id={top['accommodation_type_id']} ({top['accommodation_type']!r})"
    )
    assert any(word in top["amenity"] for word in ("grill", "barbecue", "fire")), (
        f"expected a grill-shaped amenity match, got {top['amenity']!r}"
    )
