"""amenities → subject_vectors, with category + aliases

Revision ID: 023_subject_vectors
Revises: 022_not_included_amenities
Create Date: 2026-09-02

The amenity dictionary becomes a general subject dictionary: it now holds
rules as well as amenities (`category`), and carries the surface forms that
resolve to each subject (`aliases`, GIN indexed, `aliases[1] = name`).

Views `campsites_with_amenity_names` and `accommodation_types_with_amenity_names`
need no rebuild: Postgres stores view definitions as parse trees keyed on OIDs,
so they follow the table rename. Only *adding* a column to campsites /
accommodation_types forces the DROP/CREATE dance that 007-009 and 022 do.

The HNSW index is recreated with vector_ip_ops rather than vector_cosine_ops:
every query in this repo ranks with `<#>` (negative inner product), so the
cosine index was never used.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "023_subject_vectors"
down_revision: Union[str, Sequence[str], None] = "022_not_included_amenities"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE amenities RENAME TO subject_vectors")
    op.execute("ALTER INDEX IF EXISTS amenities_pkey RENAME TO subject_vectors_pkey")
    op.execute(
        "ALTER INDEX IF EXISTS amenities_name_key RENAME TO subject_vectors_name_key"
    )

    op.execute(
        "ALTER TABLE subject_vectors "
        "ADD COLUMN category SMALLINT NOT NULL DEFAULT 1, "
        "ADD COLUMN aliases TEXT[] NOT NULL DEFAULT '{}'"
    )
    # Every pre-existing row is an amenity whose canonical name is its only alias.
    op.execute("UPDATE subject_vectors SET aliases = ARRAY[name]")

    op.execute(
        "ALTER TABLE subject_vectors "
        "ADD CONSTRAINT subject_vectors_category_check CHECK (category IN (1, 2))"
    )
    op.execute(
        "ALTER TABLE subject_vectors "
        "ADD CONSTRAINT subject_vectors_canonical_alias CHECK (aliases[1] = name)"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS subject_vectors_aliases_gin_idx "
        "ON subject_vectors USING gin (aliases)"
    )
    op.execute("DROP INDEX IF EXISTS amenity_embedding_idx")
    op.execute(
        "CREATE INDEX IF NOT EXISTS subject_vectors_embedding_idx "
        "ON subject_vectors USING hnsw (embedding vector_ip_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS subject_vectors_embedding_idx")
    op.execute("DROP INDEX IF EXISTS subject_vectors_aliases_gin_idx")
    op.execute(
        "ALTER TABLE subject_vectors "
        "DROP CONSTRAINT IF EXISTS subject_vectors_canonical_alias"
    )
    op.execute(
        "ALTER TABLE subject_vectors "
        "DROP CONSTRAINT IF EXISTS subject_vectors_category_check"
    )
    op.execute("ALTER TABLE subject_vectors DROP COLUMN aliases, DROP COLUMN category")

    op.execute(
        "ALTER INDEX IF EXISTS subject_vectors_name_key RENAME TO amenities_name_key"
    )
    op.execute("ALTER INDEX IF EXISTS subject_vectors_pkey RENAME TO amenities_pkey")
    op.execute("ALTER TABLE subject_vectors RENAME TO amenities")
    op.execute(
        "CREATE INDEX IF NOT EXISTS amenity_embedding_idx "
        "ON amenities USING hnsw (embedding vector_cosine_ops)"
    )
