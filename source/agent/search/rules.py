"""Official campsite rules, including a subcamp's parent rows."""

from __future__ import annotations

import os
from typing import Any

from langsmith import traceable
from pgvector.psycopg import register_vector

from db.connect import connect
from source.agent.search.embed import _query_vec_literal
from source.agent.search.sql import _attach_run_sql, _drop_embedding_input
from source.agent.timing import stage

# Site-wide rules for a candidate: this campsite and its parent. Sister
# subcamps (Akhziv north vs south) do not share each other's rows.
_OWN_OR_PARENT_RULES = (
    "(cr.campsite_id = {alias}.id OR cr.campsite_id = {alias}.parent_id)"
)


@traceable(
    name="search_campsite_rules",
    run_type="tool",
    process_inputs=_drop_embedding_input,
)
def search_campsite_rules(
    query: str,
    limit: int = 5,
    *,
    embedding: str | None = None,
    campsite_ids: list[int] | None = None,
) -> list[dict]:
    """Nearest official rules per campsite, all subject categories.

    Unlike the amenity lanes this includes polarity-false rows (dogs_allowed
    forbidden) and boolean/numeric rules. A subcamp reads its own rules and
    its parent's, never a sister's: ingest writes visitor-info onto the child,
    reviews live on the parent.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    if campsite_ids is not None and not campsite_ids:
        return []
    vec_literal = embedding or _query_vec_literal(query)
    ids = [int(x) for x in campsite_ids] if campsite_ids is not None else None
    if ids is None:
        sql = """
            SELECT cr.campsite_id, sv.name, sv.category, cr.polarity,
                   cr.qualifier, cr.qualifier_unit, cr.evidence_span,
                   cr.accommodation_type_id,
                   sv.embedding <#> %s::vector AS distance
            FROM campsite_rules cr
            JOIN subject_vectors sv ON sv.id = cr.subject_id
            WHERE sv.embedding IS NOT NULL
            ORDER BY sv.embedding <#> %s::vector
            LIMIT %s
        """
        params: list[Any] = [vec_literal, vec_literal, limit]
    else:
        sql = f"""
            SELECT s.campsite_id, x.name, x.category, x.polarity,
                   x.qualifier, x.qualifier_unit, x.evidence_span,
                   x.accommodation_type_id, x.distance
            FROM unnest(%s::bigint[]) AS s(campsite_id)
            JOIN campsites site ON site.id = s.campsite_id
            CROSS JOIN LATERAL (
                SELECT sv.name, sv.category, cr.polarity,
                       cr.qualifier, cr.qualifier_unit, cr.evidence_span,
                       cr.accommodation_type_id,
                       sv.embedding <#> %s::vector AS distance
                FROM campsite_rules cr
                JOIN subject_vectors sv ON sv.id = cr.subject_id
                WHERE sv.embedding IS NOT NULL
                  AND {_OWN_OR_PARENT_RULES.format(alias="site")}
                ORDER BY sv.embedding <#> %s::vector
                LIMIT %s
            ) x
        """
        params = [ids, vec_literal, vec_literal, limit]
    _attach_run_sql(sql, params)
    try:
        with stage("rules"):
            with connect(db_url) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        return [
            {
                "campsite_id": int(row[0]),
                "subject": row[1],
                "category": int(row[2]),
                "polarity": row[3],
                "qualifier": float(row[4]) if row[4] is not None else None,
                "qualifier_unit": int(row[5]) if row[5] is not None else None,
                "evidence_span": row[6],
                "accommodation_type_id": int(row[7]) if row[7] is not None else None,
                "distance": float(row[8]),
            }
            for row in rows
        ]
    except Exception as e:
        return [{"error": f"Error searching campsite rules: {e}"}]
