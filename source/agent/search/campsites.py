"""Campsite name lookup and the catalog list."""

from __future__ import annotations

import os
from typing import Any

from db.connect import connect
from source.agent.timing import stage

# pg_trgm: typos on a similar-length name, and a short query inside a long
# Hebrew title. 0.4 is below the default word_similarity threshold (0.6) so
# Achziv/Akhziv still ranks; LIMIT 5 is the cap.
NAME_LOOKUP_MIN_SCORE = 0.4
NAME_LOOKUP_LIMIT = 5

LOOKUP_CAMPSITE_SQL = """
SELECT id, name, english_name, booking_hotel_id, score
FROM (
    SELECT id, name, english_name, booking_hotel_id,
           GREATEST(
               similarity(%(q)s, name),
               similarity(%(q)s, COALESCE(english_name, '')),
               word_similarity(%(q)s, name),
               word_similarity(%(q)s, COALESCE(english_name, ''))
           ) AS score
    FROM campsites
) ranked
WHERE score >= %(min_score)s
ORDER BY score DESC, id
LIMIT %(limit)s
"""


def lookup_campsite_by_name(name: str) -> list[dict]:
    """Resolve a user-named park to campsite id(s). Not a catalog dump.

    Ranks the query against Hebrew `name` and stored `english_name` with
    pg_trgm (`similarity` + `word_similarity`). Discovery fills English;
    there is no alias list.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    query = (name or "").strip()
    if not query:
        return []
    try:
        with stage("sql"):
            with connect(db_url) as conn:
                with conn.cursor() as cur:
                    return match_campsites_by_name(cur, query)
    except Exception as e:
        return [{"error": f"Error looking up campsite: {e}"}]


def match_campsites_by_name(cur: Any, query: str) -> list[dict]:
    """pg_trgm rank of `query` vs `campsites.name` / `english_name`."""
    cur.execute(
        LOOKUP_CAMPSITE_SQL,
        {
            "q": query,
            "min_score": NAME_LOOKUP_MIN_SCORE,
            "limit": NAME_LOOKUP_LIMIT,
        },
    )
    return [
        {
            "id": int(row[0]),
            "name": row[1],
            "hotel_id": int(row[0]),
            "booking_hotel_id": row[3],
        }
        for row in cur.fetchall()
    ]


def search_campsites(numeric_constraints):
    """
    List campsites from the 'campsites' table (id, name, url).
    Numeric filters (price / ride time) are not on this table yet;
    they will come from availability data later. `numeric_constraints`
    is accepted for API compatibility with the planner node.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return "Error: DATABASE_URL not configured"
    _ = numeric_constraints  # reserved for future availability filters
    sql = """
        SELECT id, name, url
        FROM campsites
        ORDER BY id
        LIMIT 50
    """
    try:
        with connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                if not rows:
                    return "No campsites found"
                return [
                    {"id": row[0], "name": row[1], "url": row[2]}
                    for row in rows
                ]
    except Exception as e:
        return f"Error during search_campsites: {e}"
