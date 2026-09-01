"""claims.polarity text → is_positive bool

Revision ID: 019_claim_is_positive
Revises: 018_google_place_id
Create Date: 2026-09-01
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "019_claim_is_positive"
down_revision: Union[str, Sequence[str], None] = "018_google_place_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VIEW_NAME = "claims_with_reviews"

CREATE_VIEW_SQL = """
CREATE OR REPLACE VIEW claims_with_reviews AS
SELECT
  c.id,
  c.review_id,
  c.campsite_id,
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
"""

PREV_VIEW_SQL = """
CREATE OR REPLACE VIEW claims_with_reviews AS
SELECT
  c.id,
  c.review_id,
  c.campsite_id,
  c.claim,
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
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.add_column("claims", sa.Column("is_positive", sa.Boolean(), nullable=True))
    op.execute(
        """
        UPDATE claims
        SET is_positive = CASE
            WHEN polarity = 'positive' THEN TRUE
            WHEN polarity = 'negative' THEN FALSE
            ELSE NULL
        END
        """
    )
    op.drop_column("claims", "polarity")
    op.execute(CREATE_VIEW_SQL)


def downgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.add_column("claims", sa.Column("polarity", sa.Text(), nullable=True))
    op.execute(
        """
        UPDATE claims
        SET polarity = CASE
            WHEN is_positive IS TRUE THEN 'positive'
            WHEN is_positive IS FALSE THEN 'negative'
            ELSE NULL
        END
        """
    )
    op.drop_column("claims", "is_positive")
    op.execute(PREV_VIEW_SQL)
