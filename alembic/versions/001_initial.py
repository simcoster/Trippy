"""initial schema: campsites, claims, accommodation_types, availability

Revision ID: 001_initial
Revises:
Create Date: 2026-08-24

Nuke-and-pave locally when iterating:
  docker compose down -v && docker compose up -d db
  alembic upgrade head
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "001_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "campsites",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("booking_hotel_id", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("url"),
        sa.UniqueConstraint("booking_hotel_id"),
    )

    op.create_table(
        "claims",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("campsite_id", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("review_author", sa.Text(), nullable=True),
        sa.Column("review_date", sa.Text(), nullable=True),
        sa.Column("lang", sa.Text(), nullable=True),
        sa.Column("claim_he", sa.Text(), nullable=True),
        sa.Column("claim_en", sa.Text(), nullable=True),
        sa.Column("evidence_span", sa.Text(), nullable=True),
        sa.Column("polarity", sa.Text(), nullable=True),
        sa.Column("severity", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("claim_uid", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("claim_uid"),
    )
    op.create_index("claim_campsite_idx", "claims", ["campsite_id"])
    op.execute(
        "CREATE INDEX IF NOT EXISTS claim_embedding_idx "
        "ON claims USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "accommodation_types",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "availability",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.BigInteger(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("accommodation_type_id", sa.BigInteger(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("adults_no", sa.Integer(), nullable=False),
        sa.Column("room_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "scraped_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["site_id"], ["campsites.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["accommodation_type_id"],
            ["accommodation_types.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "site_id",
            "start_date",
            "end_date",
            "accommodation_type_id",
            "adults_no",
            name="availability_unique_slot",
        ),
    )
    op.create_index(
        "availability_site_dates_idx",
        "availability",
        ["site_id", "start_date", "end_date"],
    )


def downgrade() -> None:
    op.drop_index("availability_site_dates_idx", table_name="availability")
    op.drop_table("availability")
    op.drop_table("accommodation_types")
    op.execute("DROP INDEX IF EXISTS claim_embedding_idx")
    op.drop_index("claim_campsite_idx", table_name="claims")
    op.drop_table("claims")
    op.drop_table("campsites")
