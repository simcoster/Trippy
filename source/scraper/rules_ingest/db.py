"""Postgres writes for site-level campsite rules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

from db.models import QualifierUnit, SubjectCategory
from source.scraper.subjects.resolve import ensure_table_name

UPSERT_RULE_TEMPLATE = """
INSERT INTO {table} (
    campsite_id, accommodation_type_id, subject_id,
    polarity, qualifier, qualifier_unit,
    evidence_span, source_url, confidence
)
VALUES (
    %(campsite_id)s, %(accommodation_type_id)s, %(subject_id)s,
    %(polarity)s, %(qualifier)s, %(qualifier_unit)s,
    %(evidence_span)s, %(source_url)s, %(confidence)s
)
ON CONFLICT (campsite_id, accommodation_type_id, subject_id) DO UPDATE
SET polarity = EXCLUDED.polarity,
    qualifier = EXCLUDED.qualifier,
    qualifier_unit = EXCLUDED.qualifier_unit,
    evidence_span = EXCLUDED.evidence_span,
    source_url = EXCLUDED.source_url,
    confidence = EXCLUDED.confidence,
    updated_at = now()
"""

# Site-level amenities, split by polarity, mirrored into the JSONB arrays the
# planner's site lane reads (source/agent/search.py:search_site_amenities).
SYNC_AMENITY_IDS_TEMPLATE = """
UPDATE campsites
SET amenities = %(amenities)s::jsonb,
    not_included_amenities = %(not_included)s::jsonb
WHERE id = %(campsite_id)s
"""

SELECT_SITE_AMENITY_IDS_TEMPLATE = """
SELECT cr.subject_id, cr.polarity
FROM {rules_table} cr
JOIN {subjects_table} sv ON sv.id = cr.subject_id
WHERE cr.campsite_id = %(campsite_id)s
  AND cr.accommodation_type_id IS NULL
  AND sv.category = %(amenity)s
ORDER BY sv.name
"""


@dataclass(frozen=True)
class ResolvedRule:
    """A statement with its subject already resolved to a subject_vectors id."""

    subject_id: int
    polarity: bool | None = None
    qualifier: Decimal | None = None
    qualifier_unit: int = int(QualifierUnit.NONE)
    evidence_span: str | None = None
    source_url: str | None = None
    confidence: float | None = None


def upsert_campsite_rules(
    cur,
    *,
    campsite_id: int,
    rules: list[ResolvedRule],
    accommodation_type_id: int | None = None,
    table: str = "campsite_rules",
) -> int:
    """Write rules for one scope. Returns the number of statements written.

    Idempotent on (campsite_id, accommodation_type_id, subject_id), which is a
    UNIQUE NULLS NOT DISTINCT constraint — without that, site-level rows (NULL
    accommodation_type_id) would duplicate on every re-ingest.

    `table` is injected so an ingestion experiment can write to a `test_*` copy
    instead of the production table.
    """
    # The name goes into SQL, so it is never free text.
    sql = UPSERT_RULE_TEMPLATE.format(table=ensure_table_name(table))
    written = 0
    seen: dict[int, ResolvedRule] = {}
    for rule in rules:
        # The unique key allows one row per subject per scope. A repeat means two
        # statements resolved to one subject, which is usually an over-merge —
        # and if they disagree, the surviving row may now say the opposite of
        # the source, so it is called out rather than quietly skipped.
        first = seen.get(rule.subject_id)
        if first is not None:
            conflict = (
                first.polarity != rule.polarity or first.qualifier != rule.qualifier
            )
            label = "CONFLICTING" if conflict else "duplicate"
            print(
                f"    dropping {label} statement for subject {rule.subject_id}"
                f" (kept polarity={first.polarity} qualifier={first.qualifier},"
                f" dropped polarity={rule.polarity} qualifier={rule.qualifier})"
            )
            continue
        seen[rule.subject_id] = rule
        cur.execute(
            sql,
            {
                "campsite_id": campsite_id,
                "accommodation_type_id": accommodation_type_id,
                "subject_id": rule.subject_id,
                "polarity": rule.polarity,
                "qualifier": rule.qualifier,
                "qualifier_unit": int(rule.qualifier_unit),
                "evidence_span": rule.evidence_span,
                "source_url": rule.source_url,
                "confidence": rule.confidence,
            },
        )
        written += 1
    return written


def sync_campsite_amenity_ids(
    cur,
    *,
    campsite_id: int,
    rules_table: str = "campsite_rules",
    subjects_table: str = "subject_vectors",
) -> tuple[int, int]:
    """Mirror site-level amenity rules into campsites.amenities.

    The planner's site lane reads the JSONB arrays, not campsite_rules, so this
    is what makes the ingested amenities searchable today. Returns
    (included, not_included) counts. See docs/design.md — the duplication is
    deliberate and temporary.
    """
    cur.execute(
        SELECT_SITE_AMENITY_IDS_TEMPLATE.format(
            rules_table=ensure_table_name(rules_table),
            subjects_table=ensure_table_name(subjects_table),
        ),
        {"campsite_id": campsite_id, "amenity": int(SubjectCategory.AMENITY)},
    )
    included: list[int] = []
    not_included: list[int] = []
    for subject_id, polarity in cur.fetchall():
        # polarity None means a bare quantity, which still describes something
        # the site has; only an explicit False is a negative.
        (not_included if polarity is False else included).append(int(subject_id))

    cur.execute(
        SYNC_AMENITY_IDS_TEMPLATE,
        {
            "campsite_id": campsite_id,
            "amenities": json.dumps(included),
            "not_included": json.dumps(not_included),
        },
    )
    return len(included), len(not_included)
