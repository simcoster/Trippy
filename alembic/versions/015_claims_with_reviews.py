"""view claims_with_reviews

Revision ID: 015_claims_with_reviews
Revises: 014_reviews_and_claims
Create Date: 2026-09-01
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "015_claims_with_reviews"
down_revision: Union[str, Sequence[str], None] = "014_reviews_and_claims"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VIEW_NAME = "claims_with_reviews"

CREATE_VIEW_SQL = """
CREATE OR REPLACE VIEW claims_with_reviews AS
SELECT
  c.id,
  c.review_id,
  c.campsite_id,
  c.claim_en,
  c.claim_he,
  c.evidence_span,
  c.polarity,
  c.confidence,
  r.source AS review_source,
  r.author AS review_author,
  r.rating AS review_rating,
  r.text AS review_text,
  r.published_at AS review_published_at
FROM claims c
JOIN reviews r ON r.id = c.review_id
"""


def upgrade() -> None:
    op.execute(CREATE_VIEW_SQL)


def downgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
