"""Postgres read/write for accommodation amenity enrichment."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pgvector.psycopg import register_vector

from db.models import SubjectCategory

from .html_parse import MAX_IMAGE_URLS
from .llm import EmbeddingLLMClient, LlmUsage

if TYPE_CHECKING:
    from source.scraper.subjects.llm import SubjectAdjudicatorLLMClient
    from source.scraper.subjects.resolve import SubjectRef


def load_types_with_amenities(conn, hotel_id: int) -> set[str]:
    """Names of accommodation types that already have amenity ids + description.

    Reads `campsite_rules` rather than the old `accommodation_types.amenities`
    JSONB, which migration 027 dropped. A type counts as enriched when it has at
    least one rule row of its own, whatever the polarity — a tooltip that only
    yielded "bring your own towels" was still read.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT at.name
            FROM accommodation_types at
            WHERE at.hotel_id = %s
              AND at.description IS NOT NULL
              AND btrim(at.description) <> ''
              AND EXISTS (
                  SELECT 1 FROM campsite_rules cr
                  WHERE cr.accommodation_type_id = at.id
              )
            """,
            (hotel_id,),
        )
        return {row[0] for row in cur.fetchall()}


def ensure_amenities(
    conn,
    embedder: EmbeddingLLMClient,
    names: list[str],
    *,
    adjudicator: SubjectAdjudicatorLLMClient | None = None,
    contexts: dict[str, str] | None = None,
    cache: dict[str, SubjectRef] | None = None,
    usage: LlmUsage | None = None,
) -> dict[str, int]:
    """Return amenity name → subject_vectors id.

    Exact names are resolved in one batched query; anything missing goes
    through `resolve_subject`, so a new surface form attaches as an alias to an
    existing subject rather than forking a second row with its own vector.
    Names that cannot be phrased positively are dropped and simply absent from
    the returned mapping.

    `contexts` maps a name to the tooltip it was read from. It is what lets the
    sameness judge tell a room's own `bathroom` from a site's communal
    `toilets` — the names alone give it nothing to work with.
    """
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
            "SELECT id, name FROM subject_vectors WHERE name = ANY(%s)",
            (unique,),
        )
        mapping = {row[1]: int(row[0]) for row in cur.fetchall()}

    missing = [n for n in unique if n not in mapping]
    if not missing:
        return mapping

    # Imported here, not at module scope: source.scraper.subjects.llm reaches
    # back into this package's llm module, and a top-level import would make
    # that a cycle.
    from source.scraper.subjects.llm import SubjectAdjudicatorLLMClient
    from source.scraper.subjects.resolve import resolve_subject

    print(f"    resolving {len(missing)} unseen amenity name(s)")
    resolver = adjudicator or SubjectAdjudicatorLLMClient()
    shared_cache = cache if cache is not None else {}
    for name in missing:
        ref = resolve_subject(
            conn,
            name,
            embedder=embedder,
            adjudicator=resolver,
            # This path only ever handles amenities, so a rule can never be a
            # candidate however close its vector sits.
            category=int(SubjectCategory.AMENITY),
            context=(contexts or {}).get(name),
            cache=shared_cache,
            usage=usage,
        )
        if ref is not None:
            mapping[name] = ref.id
    return mapping


UPSERT_UNIT_AMENITY = """
INSERT INTO campsite_rules
    (campsite_id, accommodation_type_id, subject_id, polarity)
VALUES (%(campsite_id)s, %(accommodation_type_id)s, %(subject_id)s, %(polarity)s)
ON CONFLICT ON CONSTRAINT campsite_rules_scope_subject_key DO UPDATE
SET polarity = EXCLUDED.polarity,
    updated_at = now()
"""


def write_unit_amenities(
    cur,
    *,
    campsite_id: int,
    accommodation_type_id: int,
    amenity_ids: list[int],
    not_included_ids: list[int],
) -> int:
    """Per-unit amenities as `campsite_rules` rows scoped to one type.

    Replaces the `accommodation_types.amenities` / `not_included_amenities`
    JSONB arrays, which migration 027 dropped. The two arrays were only ever a
    polarity: provided, or explicitly not provided. Nothing else about them is
    lost, and per-unit rules now live beside site-level ones instead of in a
    parallel shape the planner had to query differently.

    A subject the extractor put in both lists for one unit collides on
    `campsite_rules_scope_subject_key`; the later write wins, which keeps the
    not-included reading — the stricter claim, and the one the tooltip was
    usually being explicit about.
    """
    written = 0
    for subject_id, polarity in (
        *((sid, True) for sid in amenity_ids),
        *((sid, False) for sid in not_included_ids),
    ):
        cur.execute(
            UPSERT_UNIT_AMENITY,
            {
                "campsite_id": campsite_id,
                "accommodation_type_id": accommodation_type_id,
                "subject_id": subject_id,
                "polarity": polarity,
            },
        )
        written += 1
    return written


def update_accommodation_type_details(
    cur,
    *,
    accommodation_type_id: int,
    description: str,
    details: dict[str, Any],
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
