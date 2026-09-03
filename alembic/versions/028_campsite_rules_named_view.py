"""campsite_rules_with_names — every id resolved to the thing it names

Revision ID: 028_campsite_rules_named_view
Revises: 027_drop_amenities_jsonb
Create Date: 2026-09-03

`campsite_rules` is four foreign keys and two enums encoded as small integers, so
reading a row by eye means four lookups:

    campsite_id=1  subject_id=417  polarity=t  qualifier=7  qualifier_unit=1

This view answers the same row as a sentence. It replaces each id with the name
it points at, and decodes the two integer enums — `subject_vectors.category` and
`qualifier_unit` — because leaving those as 1 and 2 would defeat the point.

`accommodation_type` is NULL exactly when the rule is site-wide, which is what a
NULL `accommodation_type_id` means; that is worth seeing rather than hiding
behind a placeholder, so the join is LEFT and the NULL is left alone.

This replaces the two `*_with_amenity_names` views dropped in `027`. Those read
the JSONB arrays and covered amenities only; `campsite_rules` now holds rules and
amenities, site-level and per-unit, so one view covers what both used to.

The enum text comes from `db.models.SubjectCategory` / `QualifierUnit`. A value
added there without being added here shows up as its number rather than being
silently dropped or mislabelled.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "028_campsite_rules_named_view"
down_revision: Union[str, None] = "027_drop_amenities_jsonb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


VIEW = """
CREATE VIEW campsite_rules_with_names AS
SELECT
    cr.id,
    c.name  AS campsite,
    at.name AS accommodation_type,   -- NULL means the rule is site-wide
    sv.name AS subject,
    CASE sv.category
        WHEN 1 THEN 'amenity'
        WHEN 2 THEN 'rule'
        ELSE sv.category::text
    END AS category,
    cr.polarity,
    cr.qualifier,
    CASE cr.qualifier_unit
        WHEN 0 THEN 'none'
        WHEN 1 THEN 'count'
        WHEN 2 THEN 'hour_of_day'
        WHEN 3 THEN 'nights'
        WHEN 4 THEN 'days'
        WHEN 5 THEN 'years'
        WHEN 6 THEN 'ils'
        WHEN 7 THEN 'meters'
        WHEN 8 THEN 'percent'
        ELSE cr.qualifier_unit::text
    END AS qualifier_unit,
    cr.evidence_span,
    cr.confidence,
    cr.source_url,
    cr.updated_at
FROM campsite_rules cr
JOIN campsites c        ON c.id  = cr.campsite_id
JOIN subject_vectors sv ON sv.id = cr.subject_id
LEFT JOIN accommodation_types at ON at.id = cr.accommodation_type_id
"""


def upgrade() -> None:
    op.execute(VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS campsite_rules_with_names")
