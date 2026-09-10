"""campsites.english_name — guest-facing English for named-site lookup

Revision ID: 037_campsite_english_name
Revises: 036_claims_notes
Create Date: 2026-09-10

The extractor emits English (`Achziv`, `Horashat Tal`). Hebrew `campsites.name`
does not match those, and a code alias list cannot. Discovery asks the 235B
once for every Hebrew name and stores the English here. Lookup scores the
query against `name` and `english_name` with pg_trgm (`similarity` /
`word_similarity`), not ILIKE or a code alias list.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "037_campsite_english_name"
down_revision: Union[str, Sequence[str], None] = "036_claims_notes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "campsites",
        sa.Column("english_name", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("campsites", "english_name")
