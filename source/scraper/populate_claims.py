"""Classify stored Google reviews and write claims.

Reads `reviews` where `is_relevant IS NULL`. Visit gate, split, embed —
same as the old scrape-reviews LLM path. Does not call Google.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from source.scraper.amenity_enrichment.llm import (
    ClaimsEmbeddingLLMClient,
    LlmUsage,
    make_nebius_openai_client,
    record_scrape_cost,
)
from source.scraper.populate_reviews_and_claims import (
    CONFIG_PATH,
    DELETE_CLAIMS_FOR_REVIEW_SQL,
    MIN_CONFIDENCE,
    SKIP_REASON_NOT_PERSONAL,
    database_url,
    judge_personal_visit,
    load_config,
    log,
    lookup_campsite_id,
    replace_claims,
    set_review_skip,
    split_one_review,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

SELECT_UNCLASSIFIED_SQL = """
SELECT r.id, r.campsite_id, r.source, r.author, r.rating, r.text,
       r.published_at, r.is_relevant, c.name
FROM reviews r
JOIN campsites c ON c.id = r.campsite_id
WHERE r.is_relevant IS NULL
  AND btrim(r.text) <> ''
  AND (%(campsite_id)s::bigint IS NULL OR r.campsite_id = %(campsite_id)s)
ORDER BY r.campsite_id, r.id
"""

MARK_EMPTY_IRRELEVANT_SQL = """
UPDATE reviews
SET is_relevant = FALSE
WHERE is_relevant IS NULL
  AND btrim(text) = ''
  AND (%(campsite_id)s::bigint IS NULL OR campsite_id = %(campsite_id)s)
"""


def fetch_unclassified_reviews(
    conn, *, campsite_id: int | None = None
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(SELECT_UNCLASSIFIED_SQL, {"campsite_id": campsite_id})
        return [
            {
                "id": int(row[0]),
                "campsite_id": int(row[1]),
                "source": row[2],
                "author": row[3],
                "rating": row[4],
                "text": row[5],
                "published_at": row[6],
                "is_relevant": row[7],
                "place": str(row[8] or ""),
            }
            for row in cur.fetchall()
        ]


def mark_empty_reviews_irrelevant(
    conn, *, campsite_id: int | None = None
) -> int:
    """Set is_relevant=false on unclassified empty reviews. Does not commit."""
    with conn.cursor() as cur:
        cur.execute(MARK_EMPTY_IRRELEVANT_SQL, {"campsite_id": campsite_id})
        return int(cur.rowcount or 0)


def _review_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": row["source"],
        "author": row["author"],
        "rating": row["rating"],
        "text": row["text"] or "",
        "published_at": row["published_at"],
    }


def populate_claims_for_rows(
    conn,
    rows: list[dict[str, Any]],
    *,
    chat_client: Any | None = None,
    embedder: ClaimsEmbeddingLLMClient | None = None,
    usage: LlmUsage | None = None,
) -> dict[str, int]:
    """Visit-gate, split, and embed unclassified review rows.

    Commits after each campsite so a restart skips rows already classified
    (`is_relevant` is not null).
    """
    chat = chat_client or make_nebius_openai_client()
    embed_client = embedder or ClaimsEmbeddingLLMClient(chat)
    llm_usage = usage if usage is not None else LlmUsage()
    register_vector(conn)

    by_site: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("is_relevant") is not None:
            continue
        by_site[int(row["campsite_id"])].append(row)

    n_reviews = 0
    n_claims = 0
    n_skipped = 0
    try:
        for campsite_id, site_rows in by_site.items():
            place_name = str(site_rows[0].get("place") or "")
            review_rows: list[tuple[int, list[dict]]] = []
            total = len(site_rows)
            log(
                f"Campsite {campsite_id} ({place_name}): "
                f"{total} unclassified review(s)"
            )
            with conn.cursor() as cur:
                for i, row in enumerate(site_rows, 1):
                    if row.get("is_relevant") is not None:
                        continue
                    review_id = int(row["id"])
                    review = _review_payload(row)
                    n_reviews += 1
                    if not (review.get("text") or "").strip():
                        set_review_skip(
                            cur,
                            review_id=review_id,
                            skip_reason=None,
                            skip_note=None,
                            is_relevant=False,
                        )
                        n_skipped += 1
                        log(
                            f"  review {i}/{total} id={review_id}: "
                            f"empty text, is_relevant=false"
                        )
                        continue
                    personal, skip_note = judge_personal_visit(
                        chat, review, place=place_name, usage=llm_usage
                    )
                    if not personal:
                        set_review_skip(
                            cur,
                            review_id=review_id,
                            skip_reason=SKIP_REASON_NOT_PERSONAL,
                            skip_note=skip_note,
                            is_relevant=False,
                        )
                        cur.execute(
                            DELETE_CLAIMS_FOR_REVIEW_SQL, {"review_id": review_id}
                        )
                        n_skipped += 1
                        log(
                            f"  review {i}/{total} id={review_id}: "
                            f"skipped {SKIP_REASON_NOT_PERSONAL}"
                            + (f" ({skip_note})" if skip_note else "")
                        )
                        continue
                    set_review_skip(
                        cur,
                        review_id=review_id,
                        skip_reason=None,
                        skip_note=None,
                        is_relevant=True,
                    )
                    log(
                        f"  review {i}/{total} id={review_id}: "
                        f"claim is relevant"
                    )
                    log(f"  splitting review {i}/{total} id={review_id}")
                    try:
                        claims = split_one_review(
                            chat, review, place=place_name, usage=llm_usage
                        )
                    except (ValueError, json.JSONDecodeError) as exc:
                        log(f"    split/parse failed: {exc}")
                        claims = []
                    log(
                        f"    {len(claims)} claim(s) kept "
                        f"(conf>={MIN_CONFIDENCE})"
                    )
                    review_rows.append((review_id, claims))

            all_claims = [c for _, claims in review_rows for c in claims]
            vectors: list[list[float]] = []
            if all_claims:
                log(f"  embedding {len(all_claims)} claim(s)")
                vectors = embed_client.embed(
                    [c["text_en"] for c in all_claims], usage=llm_usage
                )
            offset = 0
            with conn.cursor() as cur:
                for review_id, claims in review_rows:
                    chunk = vectors[offset : offset + len(claims)]
                    offset += len(claims)
                    n_claims += replace_claims(
                        cur,
                        review_id=review_id,
                        campsite_id=campsite_id,
                        claims=claims,
                        embeddings=chunk,
                    )
            conn.commit()
            log(f"  committed campsite_id={campsite_id}")
    except Exception:
        conn.rollback()
        raise

    result = {
        "reviews": n_reviews,
        "skipped": n_skipped,
        "claims": n_claims,
    }
    log(
        f"Done: {result['reviews']} review(s), "
        f"{result['skipped']} skipped, {result['claims']} claim(s)."
    )
    if llm_usage.chat_calls or llm_usage.embed_calls:
        log(llm_usage.summary(prefix=""))
    return result


def populate_claims(
    *,
    conn=None,
    campsite_id: int | None = None,
    chat_client: Any | None = None,
    embedder: ClaimsEmbeddingLLMClient | None = None,
    usage: LlmUsage | None = None,
) -> dict[str, int]:
    own_conn = conn is None
    if own_conn:
        config = load_config() if CONFIG_PATH.exists() else {}
        conn = psycopg.connect(database_url(config))
    try:
        n_empty = mark_empty_reviews_irrelevant(conn, campsite_id=campsite_id)
        if n_empty:
            log(f"Marked {n_empty} empty review(s) is_relevant=false")
        conn.commit()
        rows = fetch_unclassified_reviews(conn, campsite_id=campsite_id)
        if not rows:
            log("No unclassified reviews")
            return {"reviews": 0, "skipped": 0, "claims": 0}
        log(f"Classifying {len(rows)} unclassified review(s)")
        return populate_claims_for_rows(
            conn,
            rows,
            chat_client=chat_client,
            embedder=embedder,
            usage=usage,
        )
    finally:
        if own_conn:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visit-gate and split claims for reviews that are not yet "
            "classified (is_relevant IS NULL)."
        )
    )
    parser.add_argument("--campsite-id", type=int, default=None)
    parser.add_argument(
        "--name",
        default=None,
        help="Campsite name substring if --campsite-id is omitted",
    )
    args = parser.parse_args()
    config = load_config() if CONFIG_PATH.exists() else {}
    usage = LlmUsage()
    with psycopg.connect(database_url(config)) as conn:
        campsite_id = args.campsite_id
        if campsite_id is None and args.name:
            campsite_id, db_name = lookup_campsite_id(conn, args.name)
            log(f"Campsite {campsite_id}: {db_name}")
        populate_claims(conn=conn, campsite_id=campsite_id, usage=usage)
    if usage.chat_calls or usage.embed_calls:
        log(usage.summary(prefix="Claims populate total: "))
    written = record_scrape_cost("populate-claims", usage)
    if written:
        log(f"cost report appended to {written}")


if __name__ == "__main__":
    main()
