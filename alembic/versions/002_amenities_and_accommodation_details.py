"""amenities table + accommodation_types hotel/beds/amenities columns

Revision ID: 002_amenities
Revises: 001_initial
Create Date: 2026-08-25

Existing accommodation_types were global-by-name; they become per-hotel, so
prior rows (and dependent availability) are truncated before the NOT NULL
hotel_id FK is added.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "002_amenities"
down_revision: Union[str, Sequence[str], None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "TRUNCATE TABLE accommodation_types RESTART IDENTITY CASCADE"
    )

    op.create_table(
        "amenities",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS amenity_embedding_idx "
        "ON amenities USING hnsw (embedding vector_cosine_ops)"
    )

    op.add_column(
        "accommodation_types",
        sa.Column("hotel_id", sa.BigInteger(), nullable=False),
    )
    op.add_column(
        "accommodation_types",
        sa.Column("amenities", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "accommodation_types",
        sa.Column("max_occupancy", sa.Integer(), nullable=True),
    )
    op.add_column(
        "accommodation_types",
        sa.Column("total_beds", sa.Integer(), nullable=True),
    )
    op.add_column(
        "accommodation_types",
        sa.Column(
            "bed_configuration",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "accommodation_types_hotel_id_fkey",
        "accommodation_types",
        "campsites",
        ["hotel_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_constraint(
        "accommodation_types_name_key", "accommodation_types", type_="unique"
    )
    op.create_unique_constraint(
        "accommodation_types_hotel_id_name_key",
        "accommodation_types",
        ["hotel_id", "name"],
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS accommodation_types_amenities_gin_idx "
        "ON accommodation_types USING gin (amenities)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS accommodation_types_amenities_gin_idx")
    op.drop_constraint(
        "accommodation_types_hotel_id_name_key",
        "accommodation_types",
        type_="unique",
    )
    op.create_unique_constraint(
        "accommodation_types_name_key", "accommodation_types", ["name"]
    )
    op.drop_constraint(
        "accommodation_types_hotel_id_fkey", "accommodation_types", type_="foreignkey"
    )
    op.drop_column("accommodation_types", "bed_configuration")
    op.drop_column("accommodation_types", "total_beds")
    op.drop_column("accommodation_types", "max_occupancy")
    op.drop_column("accommodation_types", "amenities")
    op.drop_column("accommodation_types", "hotel_id")

    op.execute("DROP INDEX IF EXISTS amenity_embedding_idx")
    op.drop_table("amenities")
