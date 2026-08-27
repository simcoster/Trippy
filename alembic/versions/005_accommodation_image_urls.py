"""add accommodation_types.image_urls jsonb

Revision ID: 005_accommodation_image_urls
Revises: 004_accommodation_not_included
Create Date: 2026-08-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "005_accommodation_image_urls"
down_revision: Union[str, Sequence[str], None] = "004_accommodation_not_included"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "accommodation_types",
        sa.Column(
            "image_urls",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("accommodation_types", "image_urls")
