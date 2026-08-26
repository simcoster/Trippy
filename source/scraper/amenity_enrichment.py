"""Extract room amenities from INPA tooltips via Nebius (Qwen chat + embeddings)."""

from __future__ import annotations

import json
import os
import re
import ssl
from typing import Any

import httpx
from bs4 import BeautifulSoup
from openai import OpenAI
from pgvector.psycopg import register_vector

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
CHAT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBED_DIM = 1536  # HNSW index limit is 2000; Qwen3-Embedding supports MRL dims

EXTRACT_SYSTEM = """You are a precise JSON extraction engine.
Extract accommodation details from Hebrew raw text into structured JSON.
Rules:
- Output valid JSON only, without markdown wrappers.
- Count exact beds (e.g. double_bed, bunk_bed, single_bed).
- Convert Hebrew amenities to standardized snake_case English terms.
- Place missing/required items (like linens, towels) into "not_included".

Schema:
{
  "double_bed": int,
  "single_bed": int,
  "max_people": int,
  "amenities": list[str],
  "not_included": list[str]
}
"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


def nebius_client() -> OpenAI:
    api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("NEBULA_API_KEY")
    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY (or NEBULA_API_KEY) is required")
    return OpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=api_key,
        http_client=httpx.Client(verify=_ssl_context(), timeout=120.0),
    )


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


def parse_room_tooltips(html: str, normalize_name) -> dict[str, str]:
    """
    Map normalized room type → tooltip description text.

    Source: within each .roomcategory, `#toolTip_n_m > div.tt-desc > span`
    (also matches `.tt-desc .desc-span`).
    """
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, str] = {}
    for cat in soup.select("div.roomcategory"):
        name_el = cat.select_one(".roomname") or cat.select_one(".tt-roomname")
        desc_el = cat.select_one(".tt-desc span") or cat.select_one(".tt-desc")
        if not name_el or not desc_el:
            continue
        name = normalize_name(name_el.get_text(strip=True))
        text = desc_el.get_text(" ", strip=True)
        if name and text and name not in out:
            out[name] = text
    return out


def _parse_json_payload(raw: str) -> dict[str, Any]:
    text = _FENCE_RE.sub("", (raw or "").strip()).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}")
    return data


def extract_accommodation_details(client: OpenAI, raw_text: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": raw_text},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    data = _parse_json_payload(content)
    amenities = data.get("amenities") or []
    not_included = data.get("not_included") or []
    if not isinstance(amenities, list):
        amenities = []
    if not isinstance(not_included, list):
        not_included = []
    data["amenities"] = [
        str(a).strip() for a in amenities if str(a).strip()
    ]
    data["not_included"] = [
        str(a).strip() for a in not_included if str(a).strip()
    ]
    data["double_bed"] = int(data.get("double_bed") or 0)
    data["single_bed"] = int(data.get("single_bed") or 0)
    max_people = data.get("max_people")
    data["max_people"] = int(max_people) if max_people is not None else None
    return data


def ensure_amenities(
    conn,
    client: OpenAI,
    names: list[str],
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

    print(f"    embedding {len(missing)} new amenity name(s) via {EMBED_MODEL}")
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=missing,
        dimensions=EMBED_DIM,
    )
    by_index = {item.index: item.embedding for item in resp.data}

    with conn.cursor() as cur:
        for i, name in enumerate(missing):
            emb = by_index[i]
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
) -> None:
    double_beds = int(details.get("double_bed") or 0)
    single_beds = int(details.get("single_bed") or 0)
    bed_configuration = {
        "double_beds": double_beds,
        "single_beds": single_beds,
    }
    total_beds = double_beds + single_beds
    cur.execute(
        """
        UPDATE accommodation_types
        SET description = %(description)s,
            amenities = %(amenities)s::jsonb,
            not_included = %(not_included)s::jsonb,
            max_occupancy = %(max_occupancy)s,
            total_beds = %(total_beds)s,
            bed_configuration = %(bed_configuration)s::jsonb
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
        },
    )


def enrich_accommodation_types(
    conn,
    client: OpenAI,
    *,
    hotel_id: int,
    type_names: list[str],
    tooltips: dict[str, str],
    get_or_create_type,
) -> set[str]:
    """
    For each type name lacking amenities, parse tooltip → LLM → DB.

    Returns the set of type names successfully enriched in this call.
    """
    pending: list[tuple[str, str]] = []
    for name in type_names:
        text = (tooltips.get(name) or "").strip()
        if text:
            pending.append((name, text))
        else:
            print(f"    skip amenity enrich (no tooltip): {name}")

    if not pending:
        return set()

    descriptions = {name: text for name, text in pending}
    extractions: dict[str, dict[str, Any]] = {}
    for name, text in pending:
        print(f"    LLM extract amenities: {name}")
        try:
            extractions[name] = extract_accommodation_details(client, text)
        except Exception as exc:  # noqa: BLE001 — continue other types
            print(f"    LLM extract failed for {name!r}: {exc}")

    if not extractions:
        return set()

    amenity_names: list[str] = []
    for details in extractions.values():
        amenity_names.extend(details.get("amenities") or [])
        # Same canonical names table + embeddings: e.g. "shower" may appear in both lists.
        amenity_names.extend(details.get("not_included") or [])

    name_to_id = ensure_amenities(conn, client, amenity_names)
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
            update_accommodation_type_details(
                cur,
                accommodation_type_id=accom_id,
                description=descriptions[name],
                details=details,
                amenity_ids=amenity_ids,
                not_included_ids=not_included_ids,
            )
            enriched.add(name)
            print(
                f"    enriched {name}: "
                f"{len(amenity_ids)} amenities, "
                f"{len(not_included_ids)} not_included, "
                f"max_people={details.get('max_people')}, "
                f"beds={details.get('double_bed')}+{details.get('single_bed')}"
            )
    return enriched
