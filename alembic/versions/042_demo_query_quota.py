"""demo_query_quota — lifetime public-demo question count

Revision ID: 042_demo_query_quota
Revises: 041_availability_with_names
Create Date: 2026-09-22

One row per hashed visitor address. The count does not reset.
The raw address is not stored.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "042_demo_query_quota"
down_revision: Union[str, Sequence[str], None] = "041_availability_with_names"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "demo_query_quota",
        sa.Column("visitor_hash", sa.Text(), nullable=False),
        sa.Column("query_count", sa.Integer(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("visitor_hash"),
    )


def downgrade() -> None:
    op.drop_table("demo_query_quota")
