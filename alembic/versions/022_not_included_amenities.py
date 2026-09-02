"""rename accommodation_types.not_included → not_included_amenities; add the same to campsites

Revision ID: 022_not_included_amenities
Revises: 021_campsite_amenities
Create Date: 2026-09-02
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "022_not_included_amenities"
down_revision: Union[str, Sequence[str], None] = "021_campsite_amenities"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ACCOM_VIEW = "accommodation_types_with_amenity_names"
CAMPSITE_VIEW = "campsites_with_amenity_names"


def _accom_view_sql(not_included_col: str) -> str:
    return f"""
CREATE VIEW {ACCOM_VIEW} AS
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
  at.{not_included_col} AS not_included_ids,
  (
    SELECT COALESCE(array_agg(a.name ORDER BY t.ord), ARRAY[]::text[])
    FROM jsonb_array_elements(COALESCE(at.{not_included_col}, '[]'::jsonb))
         WITH ORDINALITY AS t(val, ord)
    LEFT JOIN amenities a ON a.id = (t.val)::bigint
  ) AS not_included_names
FROM accommodation_types at
"""


def _campsite_view_sql(*, with_not_included: bool) -> str:
    not_included_cols = (
        """,
  c.not_included_amenities AS not_included_ids,
  (
    SELECT COALESCE(array_agg(a.name ORDER BY t.ord), ARRAY[]::text[])
    FROM jsonb_array_elements(COALESCE(c.not_included_amenities, '[]'::jsonb))
         WITH ORDINALITY AS t(val, ord)
    LEFT JOIN amenities a ON a.id = (t.val)::bigint
  ) AS not_included_names"""
        if with_not_included
        else ""
    )
    return f"""
CREATE OR REPLACE VIEW {CAMPSITE_VIEW} AS
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
  ) AS amenity_names{not_included_cols}
FROM campsites c
"""


def upgrade() -> None:
    # Views pin the column name, so drop them before the rename.
    op.execute(f"DROP VIEW IF EXISTS {ACCOM_VIEW}")
    op.execute(f"DROP VIEW IF EXISTS {CAMPSITE_VIEW}")

    op.alter_column(
        "accommodation_types",
        "not_included",
        new_column_name="not_included_amenities",
    )
    # The index follows the column, but its name would still say not_included.
    op.execute(
        "ALTER INDEX IF EXISTS accommodation_types_not_included_gin_idx "
        "RENAME TO accommodation_types_not_included_amenities_gin_idx"
    )

    op.add_column(
        "campsites",
        sa.Column(
            "not_included_amenities",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS campsites_not_included_amenities_gin_idx "
        "ON campsites USING gin (not_included_amenities)"
    )

    op.execute(_accom_view_sql("not_included_amenities"))
    op.execute(_campsite_view_sql(with_not_included=True))


def downgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {ACCOM_VIEW}")
    op.execute(f"DROP VIEW IF EXISTS {CAMPSITE_VIEW}")

    op.execute("DROP INDEX IF EXISTS campsites_not_included_amenities_gin_idx")
    op.drop_column("campsites", "not_included_amenities")

    op.execute(
        "ALTER INDEX IF EXISTS accommodation_types_not_included_amenities_gin_idx "
        "RENAME TO accommodation_types_not_included_gin_idx"
    )
    op.alter_column(
        "accommodation_types",
        "not_included_amenities",
        new_column_name="not_included",
    )

    op.execute(_accom_view_sql("not_included"))
    op.execute(_campsite_view_sql(with_not_included=False))
