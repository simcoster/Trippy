"""drop availability.adults_no (and the same column on booking_page_hashes)

Revision ID: 039_drop_adults_no
Revises: 038_booking_page_hashes
Create Date: 2026-09-14

The INPA scrape always searches 1 adult. Party size for search is
accommodation_types.max_occupancy, not this column. Unique slot is
site + dates + type.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "039_drop_adults_no"
down_revision: Union[str, Sequence[str], None] = "038_booking_page_hashes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        "availability_unique_slot", "availability", type_="unique"
    )
    op.drop_column("availability", "adults_no")
    op.create_unique_constraint(
        "availability_unique_slot",
        "availability",
        ["site_id", "start_date", "end_date", "accommodation_type_id"],
    )

    op.drop_constraint(
        "booking_page_hashes_slot_key", "booking_page_hashes", type_="unique"
    )
    op.drop_column("booking_page_hashes", "adults_no")
    op.create_unique_constraint(
        "booking_page_hashes_slot_key",
        "booking_page_hashes",
        ["site_id", "start_date", "end_date"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "booking_page_hashes_slot_key", "booking_page_hashes", type_="unique"
    )
    op.add_column(
        "booking_page_hashes",
        sa.Column("adults_no", sa.Integer(), nullable=False, server_default="1"),
    )
    op.alter_column("booking_page_hashes", "adults_no", server_default=None)
    op.create_unique_constraint(
        "booking_page_hashes_slot_key",
        "booking_page_hashes",
        ["site_id", "start_date", "end_date", "adults_no"],
    )

    op.drop_constraint(
        "availability_unique_slot", "availability", type_="unique"
    )
    op.add_column(
        "availability",
        sa.Column("adults_no", sa.Integer(), nullable=False, server_default="1"),
    )
    op.alter_column("availability", "adults_no", server_default=None)
    op.create_unique_constraint(
        "availability_unique_slot",
        "availability",
        [
            "site_id",
            "start_date",
            "end_date",
            "accommodation_type_id",
            "adults_no",
        ],
    )
