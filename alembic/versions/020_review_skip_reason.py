"""reviews.skip_reason / skip_note for ingest visit gate

Revision ID: 020_review_skip_reason
Revises: 019_claim_is_positive
Create Date: 2026-09-01
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "020_review_skip_reason"
down_revision: Union[str, Sequence[str], None] = "019_claim_is_positive"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("reviews", sa.Column("skip_reason", sa.Text(), nullable=True))
    op.add_column("reviews", sa.Column("skip_note", sa.Text(), nullable=True))
    op.create_index(
        "reviews_skip_reason_idx",
        "reviews",
        ["skip_reason"],
        postgresql_where=sa.text("skip_reason IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("reviews_skip_reason_idx", table_name="reviews")
    op.drop_column("reviews", "skip_note")
    op.drop_column("reviews", "skip_reason")
