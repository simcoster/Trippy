"""subject_vectors.context — the sentence a subject was first read from

Revision ID: 026_subject_context
Revises: 025_subject_category_idx
Create Date: 2026-09-03

Names alone cannot separate a 30-stall communal toilet block from a room's own
bathroom: `toilets` and `bathroom` sit 0.868 apart and the sameness judge merged
them. Given the sentence each was read from —

    toilets   מה בחניון?: שירותים (15 תאי שירותי נשים ו- 15 תאי שירותי גברים)
    bathroom  חדר צוות: בכל חדר ... שירותים, מקלחת מים חמים

— it keeps them apart. Proven on `test_subject_vectors` first (see
docs/design.md); this promotes it.

Nullable and backfilled with NULL: rows created before this migration have no
recorded provenance, and a missing context simply means the judge decides on the
names alone, as it did before.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "026_subject_context"
down_revision: Union[str, Sequence[str], None] = "025_subject_category_idx"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("subject_vectors", sa.Column("context", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("subject_vectors", "context")
