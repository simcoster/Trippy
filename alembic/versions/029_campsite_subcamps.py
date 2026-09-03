"""campsites.parent_id + subcamp — one site described as two

Revision ID: 029_campsite_subcamps
Revises: 028_campsite_rules_named_view
Create Date: 2026-09-03

Akhziv is one page describing two separately-run subcamps, `חניון צפוני` and
`חניון דרומי`, each with its own amenity counts. `campsite_rules` is keyed
`(campsite_id, accommodation_type_id, subject_id)`, so the southern list
collides with the northern one and is dropped as CONFLICTING — every Akhziv
count stored today is a northern one.

A subcamp becomes a `campsites` row of its own, so **`campsite_rules` needs no
change at all**: two campsite ids, two sets of rows, the existing unique key.
Proven before this migration was written (docs/design.md): 94 rows written
through the production writer, 46 north + 48 south, zero cross-subcamp leakage.

    campsites
      2   חניון לילה גן לאומי אכזיב     url, booking_hotel_id, google_place_id
      ├─  … – חניון צפוני   parent_id=2, url NULL, hotel NULL, subcamp {...}
      └─  … – חניון דרומי   parent_id=2, url NULL, hotel NULL, subcamp {...}

The parent keeps everything describing the whole place and everything the
scrapers key on: the page, the booking id, the Google place — and therefore the
reviews, claims and prices, which are written against those keys and cannot be
attributed to one subcamp anyway. A guest review says "Akhziv", not "the
northern one". That asymmetry is the point of the shape, and it is why
`search_review_claims` resolves a child to its parent rather than finding no
claims at all.

`url` becomes nullable, which is what lets a child exist: both subcamps share one
page, and `url` is UNIQUE. Standard SQL NULLs are distinct, so any number of
children coexist under that index — no constraint is dropped or weakened. The
same holds for `booking_hotel_id`.

Two CHECKs keep the shape honest: a row is either a normal site or a subcamp
(never half of each), and a subcamp owns no page or booking id of its own — so a
scraper filtering `WHERE url IS NOT NULL` skips children for free, which is how
three of the four pipelines stay untouched.

Which sites are split is configuration, not detection — `config.json`'s
`subcamps` block, keyed by URL. See docs/design.md for why detecting it was
measured and rejected.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "029_campsite_subcamps"
down_revision: Union[str, None] = "028_campsite_rules_named_view"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "campsites",
        sa.Column(
            "parent_id",
            sa.BigInteger(),
            sa.ForeignKey("campsites.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    # {"heading": "חניון צפוני", "aliases": [...], "unit_name_contains": [...]}
    # — everything the ingest needs to know about this subcamp, so no subcamp
    # name ever appears in code.
    op.add_column("campsites", sa.Column("subcamp", postgresql.JSONB(), nullable=True))

    # A child has no page of its own. UNIQUE stays; NULLs are distinct.
    op.alter_column("campsites", "url", existing_type=sa.Text(), nullable=True)

    op.create_check_constraint(
        "campsites_subcamp_needs_parent",
        "campsites",
        "(parent_id IS NULL) = (subcamp IS NULL)",
    )
    op.create_check_constraint(
        "campsites_subcamp_has_no_page",
        "campsites",
        "parent_id IS NULL OR (url IS NULL AND booking_hotel_id IS NULL)",
    )
    op.create_index(
        "campsites_parent_idx",
        "campsites",
        ["parent_id"],
        postgresql_where=sa.text("parent_id IS NOT NULL"),
    )
    # One subcamp per name per parent, so re-running discovery updates the
    # child instead of adding another. Partial, because normal sites share the
    # NULL parent and their names are not unique across the table.
    op.create_index(
        "campsites_subcamp_name_key",
        "campsites",
        ["parent_id", "name"],
        unique=True,
        postgresql_where=sa.text("parent_id IS NOT NULL"),
    )


def downgrade() -> None:
    # Children cannot survive a schema with no parent link, and their rules go
    # with them through campsite_rules.campsite_id ON DELETE CASCADE.
    op.execute("DELETE FROM campsites WHERE parent_id IS NOT NULL")
    # IF EXISTS: these are partial indexes created by name, and a downgrade has
    # to work against a database upgraded before either existed.
    op.execute("DROP INDEX IF EXISTS campsites_subcamp_name_key")
    op.execute("DROP INDEX IF EXISTS campsites_parent_idx")
    op.drop_constraint("campsites_subcamp_has_no_page", "campsites", type_="check")
    op.drop_constraint("campsites_subcamp_needs_parent", "campsites", type_="check")
    op.alter_column("campsites", "url", existing_type=sa.Text(), nullable=False)
    op.drop_column("campsites", "subcamp")
    op.drop_column("campsites", "parent_id")
