"""Review-claim vector search."""

from __future__ import annotations

import os

from langchain_core.tools import StructuredTool
from langsmith import traceable
from pgvector.psycopg import register_vector

from db.connect import connect
from source.agent.constraints import claim_recency, today_il
from source.agent.search.embed import _query_vec_literal
from source.agent.search.sql import _attach_run_sql, _drop_embedding_input
from source.agent.timing import stage

# Global top-K over every claim. Used when no campsite scope is given.
_CLAIMS_GLOBAL_SQL = """
        SELECT c.campsite_id, c.claim, c.is_positive, r.published_at,
               c.embedding <#> %s::vector AS distance
        FROM claims c
        LEFT JOIN reviews r ON r.id = c.review_id
        WHERE c.claim IS NOT NULL
          AND (r.id IS NULL OR r.skip_reason IS NULL)
        ORDER BY c.embedding <#> %s::vector
        LIMIT %s
"""

# Top-`limit` per campsite. A global top-K crowds out a candidate site's best
# claim with another site's, so drive one scan per site instead of trimming
# after. LATERAL (not a window over the whole table) so each sub-select rides
# claim_campsite_idx.
#
# A subcamp reads its parent's claims. Reviews are written against the Google
# place, which the parent owns, and a guest says "Akhziv" rather than naming a
# subcamp — so claims only ever exist on the parent row (review-split and
# breadcrumb regions). Matching campsite_id exactly would make every subcamp
# look like a site nobody has reviewed, while the parent looked like a site
# with no amenities. The hit is still reported under the id the caller asked
# about, so a subcamp's claims rank against that subcamp's rules.
# Breadcrumb claims have review_id NULL; LEFT JOIN so they still retrieve.
_CLAIMS_BY_SITE_SQL = """
        SELECT s.campsite_id, x.claim, x.is_positive, x.published_at, x.distance
        FROM unnest(%s::bigint[]) AS s(campsite_id)
        JOIN campsites site ON site.id = s.campsite_id
        CROSS JOIN LATERAL (
            SELECT c.claim, c.is_positive, r.published_at,
                   c.embedding <#> %s::vector AS distance
            FROM claims c
            LEFT JOIN reviews r ON r.id = c.review_id
            WHERE c.campsite_id = COALESCE(site.parent_id, site.id)
              AND c.claim IS NOT NULL
              AND c.embedding IS NOT NULL
              AND (r.id IS NULL OR r.skip_reason IS NULL)
            ORDER BY c.embedding <#> %s::vector
            LIMIT %s
        ) AS x
        ORDER BY x.distance
"""


@traceable(
    name="search_review_claims",
    run_type="tool",
    process_inputs=_drop_embedding_input,
)
def search_review_claims(
    query: str,
    limit: int = 5,
    *,
    embedding: str | None = None,
    campsite_ids: list[int] | None = None,
) -> list[dict]:
    """Search review claims by vector similarity. Returns structured hits.

    With `campsite_ids`, returns the closest `limit` claims *per campsite*;
    without it, the global closest `limit` overall.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    if campsite_ids is not None and not campsite_ids:
        return []
    vec_literal = embedding or _query_vec_literal(query)
    if campsite_ids is None:
        sql = _CLAIMS_GLOBAL_SQL
        params: tuple = (vec_literal, vec_literal, limit)
    else:
        sql = _CLAIMS_BY_SITE_SQL
        params = (
            [int(x) for x in campsite_ids],
            vec_literal,
            vec_literal,
            limit,
        )
    _attach_run_sql(sql, list(params))
    try:
        today = today_il()
        with stage("retrieve"):
            with connect(db_url) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        hits: list[dict] = []
        for campsite_id, claim_text, is_positive, published_at, distance in rows:
            day, days_ago = claim_recency(published_at, today=today)
            hits.append(
                {
                    "claim": claim_text or "N/A",
                    "campsite_id": campsite_id,
                    "is_positive": is_positive,
                    "date": day,
                    "days_ago": days_ago,
                    "distance": float(distance),
                }
            )
        return hits
    except Exception as e:
        return [{"error": f"Error searching claims: {e}"}]


def search_claims(query: str, limit: int = 5) -> str:
    """
    Search for review claims using vector similarity.

    Args:
        query: The search query (e.g., "fit for stargazing", "has hot water")
        limit: Maximum number of results to return (default: 5)

    Returns:
        A formatted string with matching claims, their campsite IDs, and relevance scores.
    """
    hits = search_review_claims(query, limit=limit)
    if not hits:
        return f"No claims found matching: {query}"
    if len(hits) == 1 and hits[0].get("error"):
        return str(hits[0]["error"])
    return "\n---\n".join(
        f"Campsite: {h.get('campsite_id')}\n"
        f"Claim: {h.get('claim')}\n"
        f"Date: {h.get('date')} ({h.get('days_ago')} days ago)\n"
        f"Relevance: {h.get('distance', 0):.4f}\n"
        for h in hits
    )


claims_search_tool = StructuredTool.from_function(
    func=search_claims,
    name="search_claims",
    description=(
        "Search for review claims about campsites using semantic similarity. "
        "Use this when users ask about specific features, amenities, or experiences "
        "at campsites (e.g., 'has hot water', 'good for stargazing', 'clean facilities') "
        "that are not numeric (like 'price < 100', 'rating > 4.5', 'distance < 100km', etc.). "
        "Returns matching claims with campsite IDs and relevance scores."
    ),
)
