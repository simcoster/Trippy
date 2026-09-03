"""Orchestrate tooltip → extract → embed → DB for accommodation types."""

from __future__ import annotations

from typing import Any

from .db import (
    ensure_amenities,
    update_accommodation_type_details,
    write_unit_amenities,
)
from .html_parse import MAX_IMAGE_URLS
from .llm import EmbeddingLLMClient, ExtractorLLMClient, LlmUsage


def enrich_accommodation_types(
    conn,
    extractor: ExtractorLLMClient,
    embedder: EmbeddingLLMClient,
    *,
    hotel_id: int,
    type_names: list[str],
    room_media: dict[str, dict[str, Any]],
    get_or_create_type,
    usage: LlmUsage | None = None,
) -> set[str]:
    """
    For each type name lacking amenities, parse tooltip → LLM → DB.

    `room_media` maps normalized name → {description, image_urls}.
    Returns the set of type names successfully enriched in this call.
    """
    pending: list[tuple[str, str]] = []
    for name in type_names:
        text = ((room_media.get(name) or {}).get("description") or "").strip()
        if text:
            pending.append((name, text))
        else:
            print(f"    skip amenity enrich (no tooltip): {name}")

    if not pending:
        return set()

    batch_usage = LlmUsage()
    descriptions = {name: text for name, text in pending}
    extractions: dict[str, dict[str, Any]] = {}
    for name, text in pending:
        print(f"    LLM extract amenities: {name}")
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
        return set()

    amenity_names: list[str] = []
    # The tooltip a name came from, so the sameness judge can tell this room's
    # `bathroom` from the site's communal `toilets`. First tooltip wins: a name
    # seen in two rooms is the same subject either way.
    contexts: dict[str, str] = {}
    for name, details in extractions.items():
        tooltip = f"{name}: {descriptions[name]}"[:400]
        for key in ("amenities", "not_included"):
            # Same canonical names table + embeddings: e.g. "shower" may appear
            # in both lists.
            for amenity in details.get(key) or []:
                amenity_names.append(amenity)
                contexts.setdefault(amenity, tooltip)

    name_to_id = ensure_amenities(
        conn, embedder, amenity_names, contexts=contexts, usage=batch_usage
    )
    enriched: set[str] = set()

    with conn.cursor() as cur:
        for name, details in extractions.items():
            accom_id = get_or_create_type(cur, hotel_id=hotel_id, name=name)
            amenity_ids = [
                name_to_id[a]
                for a in (details.get("amenities") or [])
                if a in name_to_id
            ]
            not_included_ids = [
                name_to_id[a]
                for a in (details.get("not_included") or [])
                if a in name_to_id
            ]
            if not amenity_ids and not not_included_ids:
                print(f"    no amenity ids resolved for {name!r}; leaving empty")
                continue
            image_urls = (room_media.get(name) or {}).get("image_urls") or []
            update_accommodation_type_details(
                cur,
                accommodation_type_id=accom_id,
                description=descriptions[name],
                details=details,
                image_urls=image_urls,
            )
            # Amenities are rows now, not two JSONB arrays on the type.
            write_unit_amenities(
                cur,
                campsite_id=hotel_id,
                accommodation_type_id=accom_id,
                amenity_ids=amenity_ids,
                not_included_ids=not_included_ids,
            )
            enriched.add(name)
            print(
                f"    enriched {name}: "
                f"category={details.get('accommodation_category')}, "
                f"{len(amenity_ids)} amenities, "
                f"{len(not_included_ids)} not_included, "
                f"{len(image_urls[:MAX_IMAGE_URLS])} images, "
                f"max_people={details.get('max_people')}, "
                f"beds={details.get('double_bed')}+{details.get('single_bed')}, "
                f"room_count={details.get('room_count')}, "
                f"check_in={details.get('check_in_time')}, "
                f"check_out={details.get('check_out_time')}, "
                f"policy={details.get('policy_rules')}"
            )

    if usage is not None:
        usage.merge(batch_usage)
    print(batch_usage.summary())
    return enriched
