"""booking_page_hashes — skip availability writes when offers did not change

Revision ID: 038_booking_page_hashes
Revises: 037_campsite_english_name
Create Date: 2026-09-14

The INPA booking page is ASP.NET; a SHA of the raw HTML almost never
repeats. html_sha256 is stored so we can see that the bytes moved.
offers_sha256 is the skip key: canonical aggregated (room_type,
room_count). Same offers → skip DELETE/INSERT and the unit-match LLM.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "038_booking_page_hashes"
down_revision: Union[str, Sequence[str], None] = "037_campsite_english_name"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "booking_page_hashes",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.BigInteger(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("adults_no", sa.Integer(), nullable=False),
        sa.Column("html_sha256", sa.Text(), nullable=False),
        sa.Column("offers_sha256", sa.Text(), nullable=False),
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
            name="booking_page_hashes_site_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "site_id",
            "start_date",
            "end_date",
            "adults_no",
            name="booking_page_hashes_slot_key",
        ),
    )
    op.create_index(
        "booking_page_hashes_site_dates_idx",
        "booking_page_hashes",
        ["site_id", "start_date", "end_date"],
    )


def downgrade() -> None:
    op.drop_index(
        "booking_page_hashes_site_dates_idx", table_name="booking_page_hashes"
    )
    op.drop_table("booking_page_hashes")
