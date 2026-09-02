"""campsites.amenities jsonb + GIN, and a readable amenity-name view

Revision ID: 021_campsite_amenities
Revises: 020_review_skip_reason
Create Date: 2026-09-02
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "021_campsite_amenities"
down_revision: Union[str, Sequence[str], None] = "020_review_skip_reason"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VIEW_NAME = "campsites_with_amenity_names"

CREATE_VIEW_SQL = """
CREATE OR REPLACE VIEW campsites_with_amenity_names AS
SELECT
  c.id,
  c.name,
  c.url,
  c.booking_hotel_id,
  c.google_place_id,
  c.amenities AS amenity_ids,
  (
    SELECT COALESCE(array_agg(a.name ORDER BY t.ord), ARRAY[]::text[])
    FROM jsonb_array_elements(COALESCE(c.amenities, '[]'::jsonb))
         WITH ORDINALITY AS t(val, ord)
    LEFT JOIN amenities a ON a.id = (t.val)::bigint
  ) AS amenity_names
FROM campsites c
"""


def upgrade() -> None:
    op.add_column(
        "campsites",
        sa.Column("amenities", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS campsites_amenities_gin_idx "
        "ON campsites USING gin (amenities)"
    )
    op.execute(CREATE_VIEW_SQL)


def downgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.execute("DROP INDEX IF EXISTS campsites_amenities_gin_idx")
    op.drop_column("campsites", "amenities")
