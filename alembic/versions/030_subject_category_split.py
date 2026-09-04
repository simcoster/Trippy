"""Split rule subjects into boolean (2) and numeric (3)

Revision ID: 030_subject_category_split
Revises: 029_campsite_subcamps
Create Date: 2026-09-04

The merge judge was shown `late_check_out_end_time` next to
`late_check_out_allowed` and said they were one subject, with that exact pair
in its prompt as a "null" example; the 17:00 end time was then dropped on every
campsite as CONFLICTING. `early_arrival_fee_required` went into
`early_check_in_fee_percent` the same way. Both are a boolean and a number about
one topic — different kinds of fact that no judge should have to tell apart.

So `category` now separates them before the judge sees anything:

    1  amenity       provided / not provided        -> polarity
    2  boolean_rule  allowed / required             -> polarity
    3  numeric_rule  time, fee, age, nights, count  -> qualifier

The extractor tags each statement, and the resolver's nearest-neighbour search
is restricted to the term's category (the partial HNSW indexes from `025` serve
that; this adds the third). Measured on the same two pages in an isolated
schema: both merges gone, judge calls 32 -> 25, no new cross-kind merge
(experiments.md 2026-09-04 §7).

Existing category-2 rows are a mix of both kinds and are NOT reclassified here:
deciding by name suffix is exactly the string-list rule the project forbids.
The vocabulary is rebuilt by a re-scrape instead —
`scripts/clear_rules.py --subjects` then `just scrape-rules` — cents of LLM
time for 18 sites.

The `campsite_rules_with_names` view decodes the enum to text, so it learns the
two new names; `028` promised a value added to `SubjectCategory` would show up
as its number rather than mislabelled, and this keeps that promise.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "030_subject_category_split"
down_revision: Union[str, None] = "029_campsite_subcamps"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _view(categories: dict[int, str]) -> str:
    cases = "\n".join(f"        WHEN {n} THEN '{label}'" for n, label in categories.items())
    return f"""
CREATE VIEW campsite_rules_with_names AS
SELECT
    cr.id,
    c.name  AS campsite,
    at.name AS accommodation_type,   -- NULL means the rule is site-wide
    sv.name AS subject,
    CASE sv.category
{cases}
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


THREE = {1: "amenity", 2: "boolean_rule", 3: "numeric_rule"}
TWO = {1: "amenity", 2: "rule"}


def upgrade() -> None:
    op.drop_constraint("subject_vectors_category_check", "subject_vectors", type_="check")
    op.create_check_constraint(
        "subject_vectors_category_check", "subject_vectors", "category IN (1, 2, 3)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS subject_vectors_embedding_numeric_rule_idx "
        "ON subject_vectors USING hnsw (embedding vector_ip_ops) "
        "WHERE category = 3"
    )
    op.execute("DROP VIEW IF EXISTS campsite_rules_with_names")
    op.execute(_view(THREE))


def downgrade() -> None:
    # Fails while any category-3 row exists; clear the vocabulary first.
    op.execute("DROP VIEW IF EXISTS campsite_rules_with_names")
    op.execute(_view(TWO))
    op.execute("DROP INDEX IF EXISTS subject_vectors_embedding_numeric_rule_idx")
    op.drop_constraint("subject_vectors_category_check", "subject_vectors", type_="check")
    op.create_check_constraint(
        "subject_vectors_category_check", "subject_vectors", "category IN (1, 2)"
    )
