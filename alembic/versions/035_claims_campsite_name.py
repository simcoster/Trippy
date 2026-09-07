"""claims_with_reviews includes campsite name

Revision ID: 035_claims_campsite_name
Revises: 034_review_is_relevant
Create Date: 2026-09-07

The view is for eyeballing claims next to their review. `campsite_id` alone
means a second lookup for every row; join campsites and keep the id.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "035_claims_campsite_name"
down_revision: Union[str, Sequence[str], None] = "034_review_is_relevant"
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
  r.source AS review_source,
  r.author AS review_author,
  r.rating AS review_rating,
  r.text AS review_text,
  r.published_at AS review_published_at
FROM claims c
JOIN reviews r ON r.id = c.review_id
JOIN campsites s ON s.id = c.campsite_id
"""

PREV_VIEW_SQL = """
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


def upgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.execute(CREATE_VIEW_SQL)


def downgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.execute(PREV_VIEW_SQL)
