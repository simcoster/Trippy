"""campsites.google_place_id from legacy Places Text Search

Revision ID: 018_google_place_id
Revises: 017_drop_claim_uid
Create Date: 2026-09-01
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "018_google_place_id"
down_revision: Union[str, Sequence[str], None] = "017_drop_claim_uid"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "campsites",
        sa.Column("google_place_id", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("campsites", "google_place_id")
