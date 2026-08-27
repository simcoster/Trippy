"""add check_in/out time + policy_rules on accommodation_types

Revision ID: 007_accom_policy_times
Revises: 006_accom_amenity_names_view
Create Date: 2026-08-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "007_accom_policy_times"
down_revision: Union[str, Sequence[str], None] = "006_accom_amenity_names_view"
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

PREV_VIEW_SQL = """
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
    op.add_column(
        "accommodation_types",
        sa.Column("check_in_time", sa.Time(), nullable=True),
    )
    op.add_column(
        "accommodation_types",
        sa.Column("check_out_time", sa.Time(), nullable=True),
    )
    op.add_column(
        "accommodation_types",
        sa.Column(
            "policy_rules",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    # CREATE OR REPLACE cannot reorder/insert view columns; drop first.
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.execute(CREATE_VIEW_SQL)


def downgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
    op.drop_column("accommodation_types", "policy_rules")
    op.drop_column("accommodation_types", "check_out_time")
    op.drop_column("accommodation_types", "check_in_time")
    op.execute(PREV_VIEW_SQL)
