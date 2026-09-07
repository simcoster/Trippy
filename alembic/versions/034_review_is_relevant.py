"""reviews.is_relevant: visit-gate result, nullable until classify

Revision ID: 034_review_is_relevant
Revises: 033_extensions_schema
Create Date: 2026-09-07

scrape-reviews stores Google rows and leaves this NULL. populate-claims
writes true/false. clear-claims nulls it without deleting the review.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "034_review_is_relevant"
down_revision: Union[str, Sequence[str], None] = "033_extensions_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "reviews",
        sa.Column("is_relevant", sa.Boolean(), nullable=True),
    )
    op.create_index(
        "reviews_unclassified_idx",
        "reviews",
        ["campsite_id"],
        postgresql_where=sa.text("is_relevant IS NULL"),
    )
    op.execute(
        """
        UPDATE reviews
        SET is_relevant = FALSE
        WHERE skip_reason IS NOT NULL AND is_relevant IS NULL
        """
    )
    op.execute(
        """
        UPDATE reviews
        SET is_relevant = TRUE
        WHERE skip_reason IS NULL
          AND is_relevant IS NULL
          AND EXISTS (
              SELECT 1 FROM claims WHERE claims.review_id = reviews.id
          )
        """
    )


def downgrade() -> None:
    op.drop_index("reviews_unclassified_idx", table_name="reviews")
    op.drop_column("reviews", "is_relevant")
