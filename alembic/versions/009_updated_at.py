"""add updated_at on accommodation_types and availability

Revision ID: 009_updated_at
Revises: 008_accom_room_count
Create Date: 2026-08-30
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "009_updated_at"
down_revision: Union[str, Sequence[str], None] = "008_accom_room_count"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VIEW_NAME = "accommodation_types_with_amenity_names"

CREATE_VIEW_SQL = """
CREATE VIEW accommodation_types_with_amenity_names AS
SELECT
  at.id,
  at.hotel_id,
  at.name,
  at.description,
  at.max_occupancy,
  at.total_beds,
  at.room_count,
  at.bed_configuration,
  at.image_urls,
  at.check_in_time,
  at.check_out_time,
  at.policy_rules,
  at.updated_at,
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

PREV_VIEW_SQL = """
CREATE VIEW accommodation_types_with_amenity_names AS
SELECT
  at.id,
  at.hotel_id,
  at.name,
  at.description,
  at.max_occupancy,
  at.total_beds,
  at.room_count,
  at.bed_configuration,
  at.image_urls,
  at.check_in_time,
  at.check_out_time,
  at.policy_rules,
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
    op.add_column(
        "accommodation_types",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.add_column(
        "availability",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.execute(CREATE_VIEW_SQL)


def downgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.drop_column("availability", "updated_at")
    op.drop_column("accommodation_types", "updated_at")
    op.execute(PREV_VIEW_SQL)
