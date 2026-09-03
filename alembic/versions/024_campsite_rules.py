"""campsite_rules — subject × scope, with polarity and a numeric qualifier

Revision ID: 024_campsite_rules
Revises: 023_subject_vectors
Create Date: 2026-09-02

accommodation_type_id NULL means the rule is site-wide, which is every row the
info-site ingest writes. That makes NULLS NOT DISTINCT on the unique key load
bearing: without it Postgres treats each NULL as distinct and re-ingesting a
site duplicates every site-wide rule instead of updating it.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "024_campsite_rules"
down_revision: Union[str, Sequence[str], None] = "023_subject_vectors"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "campsite_rules",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("campsite_id", sa.BigInteger(), nullable=False),
        sa.Column("accommodation_type_id", sa.BigInteger(), nullable=True),
        sa.Column("subject_id", sa.BigInteger(), nullable=False),
        sa.Column("polarity", sa.Boolean(), nullable=True),
        sa.Column("qualifier", sa.Numeric(), nullable=True),
        sa.Column(
            "qualifier_unit", sa.SmallInteger(), nullable=False, server_default="0"
        ),
        sa.Column("evidence_span", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["campsite_id"], ["campsites.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["accommodation_type_id"], ["accommodation_types.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["subject_id"], ["subject_vectors.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # NULLS NOT DISTINCT (PG15+) is what dedupes site-wide rows, and it is the
    # ON CONFLICT target that makes re-ingest idempotent.
    op.execute(
        "ALTER TABLE campsite_rules ADD CONSTRAINT campsite_rules_scope_subject_key "
        "UNIQUE NULLS NOT DISTINCT (campsite_id, accommodation_type_id, subject_id)"
    )
    # subject_id leads: a btree on qualifier alone has no selectivity and the
    # values are not comparable across units (ILS vs hour-of-day).
    op.execute(
        "CREATE INDEX IF NOT EXISTS campsite_rules_subject_qualifier_idx "
        "ON campsite_rules (subject_id, qualifier)"
    )
    # The unique constraint already covers a campsite_id prefix; the other FK
    # needs its own index for the CASCADE and for per-unit lookups.
    op.execute(
        "CREATE INDEX IF NOT EXISTS campsite_rules_accom_idx "
        "ON campsite_rules (accommodation_type_id) "
        "WHERE accommodation_type_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS campsite_rules_accom_idx")
    op.execute("DROP INDEX IF EXISTS campsite_rules_subject_qualifier_idx")
    op.drop_table("campsite_rules")
