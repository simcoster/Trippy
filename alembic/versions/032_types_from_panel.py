"""accommodation_types come from the info page's lodging panel

Revision ID: 032_types_from_panel
Revises: 031_conflict_cases
Create Date: 2026-09-06

Types used to be spawned by the availability scrape from booking-engine unit
names. That made the booking engine the authority on what a campsite offers,
which is backwards: it shows only what is free on the nights scanned, so a
fully-booked or seasonally-closed unit is simply absent, and it states no
inventory counts. The info page's `אפשרויות לינה` panel states both.

Three columns go, and two arrive.

`check_in_time`, `check_out_time` and `policy_rules` have **no readers** — the
whole repo was searched: `source/`, `scripts/`, `alembic/` and the tests. The
only SQL that still names them is a downgrade-only string in
`027_drop_amenities_jsonb.py`. They are facts about a unit like any other and
belong in `campsite_rules`, where they carry the Hebrew sentence they were read
from; a minimum-night rule as an opaque JSONB key could not be traced to the
page. No backfill: `campsite_rules` and `subject_vectors` were empty when this
was written, and resolving a subject needs an LLM, which a migration must not
do. The next `scrape-info` re-derives them as rules.

`unit_count` is how many of this unit the site has — the `(48)` in
`בונגלו עם מזגן (48)`. It is NOT `room_count`, which counts rooms *within one
listing* (2 for `שתי חושות מחוברות`). 27 of the 68 units across the 18 sites
state one.

`aliases` mirrors `subject_vectors.aliases` (migration 023) down to the GIN
index and the `aliases[1] = name` check, so one idiom covers subjects and types
alike. The availability scrape no longer creates types; it matches a booking
name onto a scraped one and records the surface form here.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "032_types_from_panel"
down_revision: Union[str, None] = "031_conflict_cases"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("accommodation_types", "check_in_time")
    op.drop_column("accommodation_types", "check_out_time")
    op.drop_column("accommodation_types", "policy_rules")

    op.add_column(
        "accommodation_types", sa.Column("unit_count", sa.Integer(), nullable=True)
    )
    op.execute(
        "ALTER TABLE accommodation_types "
        "ADD COLUMN aliases TEXT[] NOT NULL DEFAULT '{}'"
    )
    # Every existing row's canonical name is its only alias, as in 023.
    op.execute("UPDATE accommodation_types SET aliases = ARRAY[name]")
    op.execute(
        "ALTER TABLE accommodation_types "
        "ADD CONSTRAINT accommodation_types_canonical_alias CHECK (aliases[1] = name)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS accommodation_types_aliases_gin_idx "
        "ON accommodation_types USING gin (aliases)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS accommodation_types_aliases_gin_idx")
    op.execute(
        "ALTER TABLE accommodation_types "
        "DROP CONSTRAINT IF EXISTS accommodation_types_canonical_alias"
    )
    op.drop_column("accommodation_types", "aliases")
    op.drop_column("accommodation_types", "unit_count")

    # Re-added empty. The values are not restored: they were never backfilled
    # out of `campsite_rules`, and rebuilding them would mean deciding which
    # rule is a check-in time, which is the judgement the rules pipeline exists
    # to make. Re-run the ingest instead.
    op.add_column(
        "accommodation_types", sa.Column("check_in_time", sa.Time(), nullable=True)
    )
    op.add_column(
        "accommodation_types", sa.Column("check_out_time", sa.Time(), nullable=True)
    )
    op.add_column(
        "accommodation_types",
        sa.Column("policy_rules", postgresql.JSONB(), nullable=True),
    )
