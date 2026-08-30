"""add ephemeral official-site notices table

Revision ID: 010_notices
Revises: 009_updated_at
Create Date: 2026-08-30
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "010_notices"
down_revision: Union[str, Sequence[str], None] = "009_updated_at"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "notices",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("page_url", sa.Text(), nullable=True),
        sa.Column("lang", sa.Text(), nullable=True),
        sa.Column("notice_he", sa.Text(), nullable=True),
        sa.Column("notice_en", sa.Text(), nullable=True),
        sa.Column("html_element", sa.Text(), nullable=False),
        sa.Column("html_element_sha256", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column(
            "first_seen",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_seen",
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
            name="notices_site_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "site_id",
            "html_element_sha256",
            name="notices_site_element_key",
        ),
    )
    op.create_index("notices_site_idx", "notices", ["site_id"])
    op.execute(
        "CREATE INDEX IF NOT EXISTS notice_embedding_idx "
        "ON notices USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS notice_embedding_idx")
    op.drop_index("notices_site_idx", table_name="notices")
    op.drop_table("notices")
