"""add accommodation_types.not_included jsonb amenity ids

Revision ID: 004_accommodation_not_included
Revises: 003_accommodation_description
Create Date: 2026-08-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "004_accommodation_not_included"
down_revision: Union[str, Sequence[str], None] = "003_accommodation_description"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "accommodation_types",
        sa.Column(
            "not_included",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS accommodation_types_not_included_gin_idx "
        "ON accommodation_types USING gin (not_included)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS accommodation_types_not_included_gin_idx")
    op.drop_column("accommodation_types", "not_included")
