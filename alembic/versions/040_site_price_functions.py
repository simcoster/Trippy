"""compiled per-site quote() stored after gold tests pass

Revision ID: 040_site_price_functions
Revises: 039_drop_adults_no
Create Date: 2026-09-15

One Python function per campsite, evaluated in the price sandbox. Hash skip
bumps scraped_at only when the source is unchanged.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "040_site_price_functions"
down_revision: Union[str, Sequence[str], None] = "039_drop_adults_no"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "site_price_functions",
        sa.Column("site_id", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column(
            "tests_passed",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
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
            name="site_price_functions_site_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("site_id"),
    )


def downgrade() -> None:
    op.drop_table("site_price_functions")
