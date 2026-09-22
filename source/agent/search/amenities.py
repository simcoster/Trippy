"""Stated unit amenities and site-wide amenities."""

from __future__ import annotations

import os
from typing import Any

from langsmith import traceable
from pgvector.psycopg import register_vector

from db.connect import connect
from db.models import SubjectCategory
from source.agent.search.embed import _query_vec_literal
from source.agent.search.rules import _OWN_OR_PARENT_RULES
from source.agent.search.sql import _attach_run_sql, _drop_embedding_input
from source.agent.timing import stage


@traceable(
    name="search_stated_amenities",
    run_type="tool",
    process_inputs=_drop_embedding_input,
)
def search_stated_amenities(
    query: str,
    limit: int = 5,
    *,
    embedding: str | None = None,
    accommodation_type_ids: list[int] | None = None,
) -> list[dict]:
    """Rank accommodation types by closest official amenity embedding."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    if accommodation_type_ids is not None and not accommodation_type_ids:
        return []
    vec_literal = embedding or _query_vec_literal(query)
    # Per-unit amenities are `campsite_rules` rows scoped to the type. The
    # `accommodation_types.amenities` JSONB this used to read was dropped in
    # migration 027; see docs/design.md.
    clauses = [
        "a.embedding IS NOT NULL",
        "a.category = %s",
        # An amenity array only ever held things the unit has. A NULL polarity
        # is a bare quantity, which still describes something present; only an
        # explicit false is a negative, and those were the separate
        # `not_included_amenities` array that this lane never read.
        "cr.polarity IS DISTINCT FROM false",
    ]
    params: list[Any] = [vec_literal, vec_literal, int(SubjectCategory.AMENITY)]
    if accommodation_type_ids is not None:
        clauses.append("at.id = ANY(%s)")
        params.append([int(x) for x in accommodation_type_ids])
    params.append(limit)
    sql = f"""
        SELECT at.id,
               at.name,
               at.hotel_id,
               MIN(a.embedding <#> %s::vector) AS distance,
               (array_agg(a.name ORDER BY a.embedding <#> %s::vector))[1]
                   AS matched_amenity
        FROM accommodation_types at
        JOIN campsite_rules cr ON cr.accommodation_type_id = at.id
        JOIN subject_vectors a ON a.id = cr.subject_id
        WHERE {' AND '.join(clauses)}
        GROUP BY at.id, at.name, at.hotel_id
        ORDER BY distance
        LIMIT %s
    """
    _attach_run_sql(sql, params)
    try:
        with stage("retrieve"):
            with connect(db_url) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        return [
            {
                "amenity": row[4],
                "accommodation_type_id": int(row[0]),
                "accommodation_type": row[1],
                "hotel_id": int(row[2]),
                "distance": float(row[3]),
            }
            for row in rows
        ]
    except Exception as e:
        return [{"error": f"Error searching stated amenities: {e}"}]


@traceable(
    name="search_site_amenities",
    run_type="tool",
    process_inputs=_drop_embedding_input,
)
def search_site_amenities(
    query: str,
    limit: int = 5,
    *,
    embedding: str | None = None,
    campsite_ids: list[int] | None = None,
) -> list[dict]:
    """Rank campsites by closest site-wide (communal) amenity embedding.

    A subcamp's scan includes the parent's site-wide rows, not a sister's.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    if campsite_ids is not None and not campsite_ids:
        return []
    vec_literal = embedding or _query_vec_literal(query)
    # Site-wide amenities are `campsite_rules` rows with no accommodation type.
    # `campsites.amenities` was a mirror of exactly these rows and was dropped
    # in migration 027, along with the sync step that maintained it.
    clauses = [
        "a.embedding IS NOT NULL",
        "a.category = %s",
        "cr.polarity IS DISTINCT FROM false",
    ]
    params: list[Any] = [vec_literal, vec_literal, int(SubjectCategory.AMENITY)]
    if campsite_ids is not None:
        clauses.append("c.id = ANY(%s)")
        params.append([int(x) for x in campsite_ids])
    params.append(limit)
    sql = f"""
        SELECT c.id,
               c.name,
               MIN(a.embedding <#> %s::vector) AS distance,
               (array_agg(a.name ORDER BY a.embedding <#> %s::vector))[1]
                   AS matched_amenity
        FROM campsites c
        JOIN campsite_rules cr
          ON cr.accommodation_type_id IS NULL
         AND {_OWN_OR_PARENT_RULES.format(alias="c")}
        JOIN subject_vectors a ON a.id = cr.subject_id
        WHERE {' AND '.join(clauses)}
        GROUP BY c.id, c.name
        ORDER BY distance
        LIMIT %s
    """
    _attach_run_sql(sql, params)
    try:
        with stage("retrieve"):
            with connect(db_url) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        return [
            {
                "amenity": row[3],
                "campsite_id": int(row[0]),
                "campsite": row[1],
                "distance": float(row[2]),
            }
            for row in rows
        ]
    except Exception as e:
        return [{"error": f"Error searching site amenities: {e}"}]
