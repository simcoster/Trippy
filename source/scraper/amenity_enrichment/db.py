"""Postgres read/write for accommodation amenity enrichment."""

from __future__ import annotations

import json
from typing import Any

from pgvector.psycopg import register_vector

from .html_parse import MAX_IMAGE_URLS
from .llm import EmbeddingLLMClient, LlmUsage


def load_types_with_amenities(conn, hotel_id: int) -> set[str]:
    """Names of accommodation types that already have amenity ids + description."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT name
            FROM accommodation_types
            WHERE hotel_id = %s
              AND amenities IS NOT NULL
              AND jsonb_typeof(amenities) = 'array'
              AND jsonb_array_length(amenities) > 0
              AND description IS NOT NULL
              AND btrim(description) <> ''
            """,
            (hotel_id,),
        )
        return {row[0] for row in cur.fetchall()}


def ensure_amenities(
    conn,
    embedder: EmbeddingLLMClient,
    names: list[str],
    *,
    usage: LlmUsage | None = None,
) -> dict[str, int]:
    """Return amenity name → id; insert + embed any missing names in one batch."""
    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    if not unique:
        return {}

    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name FROM amenities WHERE name = ANY(%s)",
            (unique,),
        )
        mapping = {row[1]: int(row[0]) for row in cur.fetchall()}

    missing = [n for n in unique if n not in mapping]
    if not missing:
        return mapping

    print(f"    embedding {len(missing)} new amenity name(s) via {embedder.MODEL}")
    vectors = embedder.embed(missing, usage=usage)

    with conn.cursor() as cur:
        for name, emb in zip(missing, vectors, strict=True):
            cur.execute(
                """
                INSERT INTO amenities (name, embedding)
                VALUES (%(name)s, %(embedding)s)
                ON CONFLICT (name) DO UPDATE
                SET embedding = COALESCE(amenities.embedding, EXCLUDED.embedding)
                RETURNING id, name
                """,
                {"name": name, "embedding": emb},
            )
            row = cur.fetchone()
            mapping[row[1]] = int(row[0])
    return mapping


def update_accommodation_type_details(
    cur,
    *,
    accommodation_type_id: int,
    description: str,
    details: dict[str, Any],
    amenity_ids: list[int],
    not_included_ids: list[int],
    image_urls: list[str] | None = None,
) -> None:
    double_beds = int(details.get("double_bed") or 0)
    single_beds = int(details.get("single_bed") or 0)
    bed_configuration = {
        "double_beds": double_beds,
        "single_beds": single_beds,
    }
    total_beds = double_beds + single_beds
    urls = [u for u in (image_urls or []) if u][:MAX_IMAGE_URLS]
    policy_rules = details.get("policy_rules")
    cur.execute(
        """
        UPDATE accommodation_types
        SET description = %(description)s,
            amenities = %(amenities)s::jsonb,
            not_included = %(not_included)s::jsonb,
            max_occupancy = %(max_occupancy)s,
            total_beds = %(total_beds)s,
            bed_configuration = %(bed_configuration)s::jsonb,
            image_urls = %(image_urls)s::jsonb,
            check_in_time = %(check_in_time)s,
            check_out_time = %(check_out_time)s,
            policy_rules = %(policy_rules)s::jsonb,
            room_count = %(room_count)s,
            updated_at = now()
        WHERE id = %(id)s
        """,
        {
            "id": accommodation_type_id,
            "description": description,
            "amenities": json.dumps(amenity_ids),
            "not_included": json.dumps(not_included_ids),
            "max_occupancy": details.get("max_people"),
            "total_beds": total_beds if total_beds > 0 else None,
            "bed_configuration": json.dumps(bed_configuration),
            "image_urls": json.dumps(urls) if urls else None,
            "check_in_time": details.get("check_in_time"),
            "check_out_time": details.get("check_out_time"),
            "policy_rules": json.dumps(policy_rules) if policy_rules else None,
            "room_count": int(details.get("room_count") or 1),
        },
    )


def fill_missing_image_urls(
    conn,
    *,
    hotel_id: int,
    room_media: dict[str, dict[str, Any]],
) -> int:
    """Set image_urls for types that exist but have none yet. Returns rows updated."""
    updated = 0
    with conn.cursor() as cur:
        for name, meta in room_media.items():
            urls = [u for u in (meta.get("image_urls") or []) if u][:MAX_IMAGE_URLS]
            if not urls:
                continue
            cur.execute(
                """
                UPDATE accommodation_types
                SET image_urls = %(image_urls)s::jsonb,
                    updated_at = now()
                WHERE hotel_id = %(hotel_id)s
                  AND name = %(name)s
                  AND (
                    image_urls IS NULL
                    OR jsonb_typeof(image_urls) <> 'array'
                    OR jsonb_array_length(image_urls) = 0
                  )
                """,
                {
                    "hotel_id": hotel_id,
                    "name": name,
                    "image_urls": json.dumps(urls),
                },
            )
            updated += cur.rowcount
    return updated
