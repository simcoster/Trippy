"""Drop the four amenities JSONB columns; campsite_rules is the only home

Revision ID: 027_drop_amenities_jsonb
Revises: 026_subject_context
Create Date: 2026-09-03

`campsites.amenities` / `not_included_amenities` were a derived mirror of
site-level `campsite_rules` rows, maintained by a sync step after every ingest,
because the planner's two amenity lanes read the JSONB rather than the table
(docs/design.md, "Amenities live in two places"). Those lanes now read
`campsite_rules`, so the mirror and the sync step both go.

`accommodation_types.amenities` / `not_included_amenities` were **not** a
mirror. Before this migration:

    accommodation_types with a non-empty amenities array   12
    campsite_rules rows with accommodation_type_id set       0

Per-unit amenities lived only in that JSONB, so this migration backfills them
into rows before dropping the columns — which is also what finally fills
`accommodation_type_id`, the "always NULL today" column docs/design.md describes.
The two arrays carried one bit between them, provided vs explicitly not
provided, and that is `campsite_rules.polarity`; nothing else is lost.

Verified before writing this, in `temp/drop_amenities_jsonb_testbed.py`: the
backfill carried all 98 ids with none dropped, and running both lanes' old
(JSONB) and new (campsite_rules) SQL over every one of the 105 subject vectors
as a query returned identical top-5 results, 105/105, on both lanes.

The two `*_with_amenity_names` views read the JSONB and are dropped with it.
They existed to make the id arrays readable by eye; the equivalent is now a
plain join, and nothing but a schema test referenced them.

Downgrade restores the columns, the views, and the data, since every id is
recoverable from the rows this migration wrote.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "027_drop_amenities_jsonb"
down_revision: Union[str, None] = "026_subject_context"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# A subject listed in both arrays for one unit would violate
# campsite_rules_scope_subject_key. The not-included pass runs second and wins,
# which keeps the stricter reading.
BACKFILL = """
INSERT INTO campsite_rules
    (campsite_id, accommodation_type_id, subject_id, polarity)
SELECT at.hotel_id, at.id, (elem.val)::bigint, :polarity
FROM accommodation_types at
CROSS JOIN LATERAL
    jsonb_array_elements(COALESCE(at.{column}, '[]'::jsonb)) AS elem(val)
JOIN subject_vectors sv ON sv.id = (elem.val)::bigint
ON CONFLICT ON CONSTRAINT campsite_rules_scope_subject_key DO UPDATE
SET polarity = EXCLUDED.polarity
"""

RESTORE = """
UPDATE {table} t
SET {column} = COALESCE(sub.ids, '[]'::jsonb)
FROM (
    SELECT {scope} AS scope_id,
           jsonb_agg(cr.subject_id ORDER BY cr.subject_id) AS ids
    FROM campsite_rules cr
    JOIN subject_vectors sv ON sv.id = cr.subject_id
    WHERE sv.category = 1
      AND cr.accommodation_type_id IS {null_test}
      AND cr.polarity IS {polarity_test}
    GROUP BY {scope}
) sub
WHERE t.id = sub.scope_id
"""

CAMPSITES_VIEW = """
CREATE VIEW campsites_with_amenity_names AS
SELECT id, name, url, booking_hotel_id, google_place_id,
       amenities AS amenity_ids,
       (SELECT COALESCE(array_agg(a.name ORDER BY t.ord), ARRAY[]::text[])
          FROM jsonb_array_elements(COALESCE(c.amenities, '[]'::jsonb))
               WITH ORDINALITY t(val, ord)
          LEFT JOIN subject_vectors a ON a.id = (t.val)::bigint) AS amenity_names,
       not_included_amenities AS not_included_ids
FROM campsites c
"""

ACCOM_VIEW = """
CREATE VIEW accommodation_types_with_amenity_names AS
SELECT id, hotel_id, name, description, max_occupancy, total_beds, room_count,
       bed_configuration, image_urls, check_in_time, check_out_time,
       policy_rules, updated_at,
       amenities AS amenity_ids,
       (SELECT COALESCE(array_agg(a.name ORDER BY t.ord), ARRAY[]::text[])
          FROM jsonb_array_elements(COALESCE(at.amenities, '[]'::jsonb))
               WITH ORDINALITY t(val, ord)
          LEFT JOIN subject_vectors a ON a.id = (t.val)::bigint) AS amenity_names,
       not_included_amenities AS not_included_ids
FROM accommodation_types at
"""


def upgrade() -> None:
    bind = op.get_bind()

    # Backfill BEFORE dropping: this is the only copy of per-unit amenities.
    for column, polarity in (("amenities", True), ("not_included_amenities", False)):
        bind.execute(
            sa.text(BACKFILL.format(column=column)), {"polarity": polarity}
        )

    op.execute("DROP VIEW IF EXISTS campsites_with_amenity_names")
    op.execute("DROP VIEW IF EXISTS accommodation_types_with_amenity_names")

    op.drop_column("campsites", "amenities")
    op.drop_column("campsites", "not_included_amenities")
    op.drop_column("accommodation_types", "amenities")
    op.drop_column("accommodation_types", "not_included_amenities")


def downgrade() -> None:
    op.add_column("campsites", sa.Column("amenities", postgresql.JSONB()))
    op.add_column(
        "campsites",
        sa.Column("not_included_amenities", postgresql.JSONB()),
    )
    op.add_column(
        "accommodation_types", sa.Column("amenities", postgresql.JSONB())
    )
    op.add_column(
        "accommodation_types",
        sa.Column("not_included_amenities", postgresql.JSONB()),
    )

    bind = op.get_bind()
    # Rebuild each array from the rows. A NULL polarity is a bare quantity,
    # which the `amenities` array counted as present; only an explicit false
    # belonged in `not_included_amenities`.
    for table, scope, null_test in (
        ("campsites", "cr.campsite_id", "NULL"),
        ("accommodation_types", "cr.accommodation_type_id", "NOT NULL"),
    ):
        for column, polarity_test in (
            ("amenities", "DISTINCT FROM false"),
            ("not_included_amenities", "false"),
        ):
            bind.execute(
                sa.text(
                    RESTORE.format(
                        table=table,
                        column=column,
                        scope=scope,
                        null_test=null_test,
                        polarity_test=polarity_test,
                    )
                )
            )

    op.execute(CAMPSITES_VIEW)
    op.execute(ACCOM_VIEW)
