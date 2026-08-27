"""view accommodation_types_with_amenity_names

Revision ID: 006_accom_amenity_names_view
Revises: 005_accommodation_image_urls
Create Date: 2026-08-27
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "006_accom_amenity_names_view"
down_revision: Union[str, Sequence[str], None] = "005_accommodation_image_urls"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VIEW_NAME = "accommodation_types_with_amenity_names"

CREATE_VIEW_SQL = """
CREATE OR REPLACE VIEW accommodation_types_with_amenity_names AS
SELECT
  at.id,
  at.hotel_id,
  at.name,
  at.description,
  at.max_occupancy,
  at.total_beds,
  at.bed_configuration,
  at.image_urls,
  at.amenities AS amenity_ids,
  (
    SELECT COALESCE(array_agg(a.name ORDER BY t.ord), ARRAY[]::text[])
    FROM jsonb_array_elements(COALESCE(at.amenities, '[]'::jsonb))
         WITH ORDINALITY AS t(val, ord)
    LEFT JOIN amenities a ON a.id = (t.val)::bigint
  ) AS amenity_names,
  at.not_included AS not_included_ids,
  (
    SELECT COALESCE(array_agg(a.name ORDER BY t.ord), ARRAY[]::text[])
    FROM jsonb_array_elements(COALESCE(at.not_included, '[]'::jsonb))
         WITH ORDINALITY AS t(val, ord)
    LEFT JOIN amenities a ON a.id = (t.val)::bigint
  ) AS not_included_names
FROM accommodation_types at
"""


def upgrade() -> None:
    op.execute(CREATE_VIEW_SQL)


def downgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
