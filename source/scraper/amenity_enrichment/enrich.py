"""Orchestrate tooltip → extract → DB for accommodation types.

Structured columns only: beds, occupancy, times, images, policy JSONB. The
unit's amenities and rules are no longer read here — they go through the site
rules pipeline (`rules_ingest.units.ingest_unit_rules`), which writes them to
`campsite_rules` with the sentence each was read from.
"""

from __future__ import annotations

from typing import Any

from .db import update_accommodation_type_details
from .html_parse import MAX_IMAGE_URLS
from .llm import ExtractorLLMClient, LlmUsage


def enrich_accommodation_types(
    conn,
    extractor: ExtractorLLMClient,
    *,
    hotel_id: int,
    type_names: list[str],
    room_media: dict[str, dict[str, Any]],
    get_or_create_type,
    usage: LlmUsage | None = None,
) -> dict[str, int]:
    """For each type name, parse tooltip → LLM → the type's own columns.

    `room_media` maps normalized name → {description, image_urls}. Returns the
    types written, name → `accommodation_types.id`, which is the scope the
    caller then ingests that unit's rules into.

    The tooltip is left on `accommodation_types.description`: the rules pass
    reads the same text, and the column is what `load_types_with_amenities`
    checks to decide a type has been read at all.
    """
    pending: list[tuple[str, str]] = []
    for name in type_names:
        text = ((room_media.get(name) or {}).get("description") or "").strip()
        if text:
            pending.append((name, text))
        else:
            print(f"    skip enrich (no tooltip): {name}")

    if not pending:
        return {}

    batch_usage = LlmUsage()
    descriptions = {name: text for name, text in pending}
    extractions: dict[str, dict[str, Any]] = {}
    for name, text in pending:
        print(f"    LLM extract unit details: {name}")
        try:
            extractions[name] = extractor.extract(
                text, type_name=name, usage=batch_usage
            )
        except Exception as exc:  # noqa: BLE001 — continue other types
            print(f"    LLM extract failed for {name!r}: {exc}")

    if not extractions:
        if usage is not None:
            usage.merge(batch_usage)
        if batch_usage.chat_calls or batch_usage.embed_calls:
            print(batch_usage.summary())
        return {}

    written: dict[str, int] = {}
    with conn.cursor() as cur:
        for name, details in extractions.items():
            accom_id = get_or_create_type(cur, hotel_id=hotel_id, name=name)
            image_urls = (room_media.get(name) or {}).get("image_urls") or []
            update_accommodation_type_details(
                cur,
                accommodation_type_id=accom_id,
                description=descriptions[name],
                details=details,
                image_urls=image_urls,
            )
            written[name] = accom_id
            print(
                f"    enriched {name}: "
                f"category={details.get('accommodation_category')}, "
                f"{len(image_urls[:MAX_IMAGE_URLS])} images, "
                f"max_people={details.get('max_people')}, "
                f"beds={details.get('double_bed')}+{details.get('single_bed')}, "
                f"room_count={details.get('room_count')}, "
                f"spans={len(details.get('consumed_spans') or [])}"
            )

    if usage is not None:
        usage.merge(batch_usage)
    print(batch_usage.summary())
    return written
