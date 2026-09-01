"""claims.claim + evidence_span; drop claim_he / claim_en

Revision ID: 016_claim_and_evidence_span
Revises: 015_claims_with_reviews
Create Date: 2026-09-01
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "016_claim_and_evidence_span"
down_revision: Union[str, Sequence[str], None] = "015_claims_with_reviews"
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

PREV_VIEW_SQL = """
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
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.drop_column("claims", "claim_he")
    op.alter_column("claims", "claim_en", new_column_name="claim")
    op.execute(CREATE_VIEW_SQL)


def downgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.alter_column("claims", "claim", new_column_name="claim_en")
    op.execute("ALTER TABLE claims ADD COLUMN claim_he TEXT")
    op.execute(PREV_VIEW_SQL)
