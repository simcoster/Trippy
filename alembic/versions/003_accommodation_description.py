"""add accommodation_types.description for raw tooltip text

Revision ID: 003_accommodation_description
Revises: 002_amenities
Create Date: 2026-08-26
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003_accommodation_description"
down_revision: Union[str, Sequence[str], None] = "002_amenities"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "accommodation_types",
        sa.Column("description", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("accommodation_types", "description")
