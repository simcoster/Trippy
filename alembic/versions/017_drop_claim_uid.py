"""drop unused claims.claim_uid

Revision ID: 017_drop_claim_uid
Revises: 016_claim_and_evidence_span
Create Date: 2026-09-01
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "017_drop_claim_uid"
down_revision: Union[str, Sequence[str], None] = "016_claim_and_evidence_span"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("claims", "claim_uid")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE claims ADD COLUMN claim_uid TEXT"
    )
    op.execute(
        "UPDATE claims SET claim_uid = id::text WHERE claim_uid IS NULL"
    )
    op.alter_column("claims", "claim_uid", nullable=False)
    op.create_unique_constraint("claims_claim_uid_key", "claims", ["claim_uid"])
