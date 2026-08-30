"""add published rate-card list_prices table

Revision ID: 011_list_prices
Revises: 010_notices
Create Date: 2026-08-30
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "011_list_prices"
down_revision: Union[str, Sequence[str], None] = "010_notices"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "list_prices",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.BigInteger(), nullable=False),
        sa.Column("accommodation_type_id", sa.BigInteger(), nullable=False),
        sa.Column("guest_type", sa.Text(), nullable=False),
        sa.Column("rate_period", sa.Text(), nullable=False),
        sa.Column("rate_class", sa.Text(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("currency", sa.Text(), nullable=False, server_default="ILS"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("raw_label", sa.Text(), nullable=False),
        sa.Column(
            "scraped_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["site_id"],
            ["campsites.id"],
            name="list_prices_site_id_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["accommodation_type_id"],
            ["accommodation_types.id"],
            name="list_prices_accommodation_type_id_fkey",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "site_id",
            "accommodation_type_id",
            "guest_type",
            "rate_period",
            "rate_class",
            name="list_prices_unique_rate",
        ),
    )
    op.create_index("list_prices_site_idx", "list_prices", ["site_id"])


def downgrade() -> None:
    op.drop_index("list_prices_site_idx", table_name="list_prices")
    op.drop_table("list_prices")
