"""claims.notes and optional review_id (breadcrumb region claims)

Revision ID: 036_claims_notes
Revises: 035_claims_campsite_name
Create Date: 2026-09-09

scrape-info writes parks.org.il #breadcrumbs as claims with no review row.
`notes` marks those rows (`no review, region by breadcrumbs`); review-split
claims leave it NULL. `review_id` is nullable so they can exist. The
eyeball view left-joins reviews so they still show.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "036_claims_notes"
down_revision: Union[str, Sequence[str], None] = "035_claims_campsite_name"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VIEW_NAME = "claims_with_reviews"

CREATE_VIEW_SQL = """
CREATE OR REPLACE VIEW claims_with_reviews AS
SELECT
  c.id,
  c.review_id,
  c.campsite_id,
  s.name AS campsite,
  c.claim,
  c.evidence_span,
  c.is_positive,
  c.confidence,
  c.notes,
  r.source AS review_source,
  r.author AS review_author,
  r.rating AS review_rating,
  r.text AS review_text,
  r.published_at AS review_published_at
FROM claims c
LEFT JOIN reviews r ON r.id = c.review_id
JOIN campsites s ON s.id = c.campsite_id
"""

PREV_VIEW_SQL = """
CREATE OR REPLACE VIEW claims_with_reviews AS
SELECT
  c.id,
  c.review_id,
  c.campsite_id,
  s.name AS campsite,
  c.claim,
  c.evidence_span,
  c.is_positive,
  c.confidence,
  r.source AS review_source,
  r.author AS review_author,
  r.rating AS review_rating,
  r.text AS review_text,
  r.published_at AS review_published_at
FROM claims c
JOIN reviews r ON r.id = c.review_id
JOIN campsites s ON s.id = c.campsite_id
"""


def upgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.add_column("claims", sa.Column("notes", sa.Text(), nullable=True))
    op.alter_column(
        "claims",
        "review_id",
        existing_type=sa.BigInteger(),
        nullable=True,
    )
    op.execute(CREATE_VIEW_SQL)


def downgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.execute("DELETE FROM claims WHERE review_id IS NULL")
    op.alter_column(
        "claims",
        "review_id",
        existing_type=sa.BigInteger(),
        nullable=False,
    )
    op.drop_column("claims", "notes")
    op.execute(PREV_VIEW_SQL)
