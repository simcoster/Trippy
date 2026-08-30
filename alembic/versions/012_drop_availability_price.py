"""drop availability.price; quotes come from list_prices

Revision ID: 012_drop_availability_price
Revises: 011_list_prices
Create Date: 2026-08-30
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "012_drop_availability_price"
down_revision: Union[str, Sequence[str], None] = "011_list_prices"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("availability", "price")


def downgrade() -> None:
    op.add_column(
        "availability",
        sa.Column("price", sa.Float(), nullable=False, server_default="0"),
    )
    op.alter_column("availability", "price", server_default=None)
