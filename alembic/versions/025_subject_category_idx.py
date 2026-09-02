"""Per-category HNSW indexes on subject_vectors

Revision ID: 025_subject_category_idx
Revises: 024_campsite_rules
Create Date: 2026-09-02

The resolver never wants a rule as a candidate for an amenity, or the reverse:
`barbecue_allowed` sits closer to `barbecue` (-0.891) than the correct
`barbecue_equipment_included` does (-0.854), so an unpartitioned nearest-
neighbour search ranks the wrong subject first.

A partial index per category makes `WHERE category = N ORDER BY embedding <#> ?`
its own index scan instead of a filter over the shared HNSW graph — the same
plan two separate tables would give, without a polymorphic
`campsite_rules.subject_id`. Two branches can be searched in one round trip:

    (SELECT ... WHERE category = 1 ORDER BY embedding <#> %s LIMIT 5)
    UNION ALL
    (SELECT ... WHERE category = 2 ORDER BY embedding <#> %s LIMIT 5)

The unpartitioned index stays: the planner's amenity lanes join on ids from a
JSONB array and never filter by category.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "025_subject_category_idx"
down_revision: Union[str, Sequence[str], None] = "024_campsite_rules"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS subject_vectors_embedding_amenity_idx "
        "ON subject_vectors USING hnsw (embedding vector_ip_ops) "
        "WHERE category = 1"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS subject_vectors_embedding_rule_idx "
        "ON subject_vectors USING hnsw (embedding vector_ip_ops) "
        "WHERE category = 2"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS subject_vectors_embedding_rule_idx")
    op.execute("DROP INDEX IF EXISTS subject_vectors_embedding_amenity_idx")
