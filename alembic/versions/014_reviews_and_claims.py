"""reviews table; drop and recreate claims with review_id FK

Revision ID: 014_reviews_and_claims
Revises: 013_info_website_names
Create Date: 2026-09-01
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "014_reviews_and_claims"
down_revision: Union[str, Sequence[str], None] = "013_info_website_names"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS claim_embedding_idx")
    op.drop_index("claim_campsite_idx", table_name="claims")
    op.drop_table("claims")

    op.create_table(
        "reviews",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("campsite_id", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False, server_default=""),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_uid", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["campsite_id"],
            ["campsites.id"],
            name="reviews_campsite_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("review_uid", name="reviews_review_uid_key"),
    )
    op.create_index("reviews_campsite_idx", "reviews", ["campsite_id"])

    op.create_table(
        "claims",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("review_id", sa.BigInteger(), nullable=False),
        sa.Column("campsite_id", sa.BigInteger(), nullable=False),
        sa.Column("claim_he", sa.Text(), nullable=True),
        sa.Column("claim_en", sa.Text(), nullable=True),
        sa.Column("evidence_span", sa.Text(), nullable=True),
        sa.Column("polarity", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("claim_uid", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.ForeignKeyConstraint(
            ["review_id"],
            ["reviews.id"],
            name="claims_review_id_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["campsite_id"],
            ["campsites.id"],
            name="claims_campsite_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("claim_uid", name="claims_claim_uid_key"),
    )
    op.create_index("claim_campsite_idx", "claims", ["campsite_id"])
    op.create_index("claim_review_idx", "claims", ["review_id"])
    op.execute(
        "CREATE INDEX IF NOT EXISTS claim_embedding_idx "
        "ON claims USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS claim_embedding_idx")
    op.drop_index("claim_review_idx", table_name="claims")
    op.drop_index("claim_campsite_idx", table_name="claims")
    op.drop_table("claims")
    op.drop_index("reviews_campsite_idx", table_name="reviews")
    op.drop_table("reviews")

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
