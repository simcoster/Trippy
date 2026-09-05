"""conflict_cases — every upsert collision, filed for review

Revision ID: 031_conflict_cases
Revises: 030_subject_category_split
Create Date: 2026-09-05

A collision (two statements from one page landing on one subject) always means
something upstream went wrong: the extractor dropped a qualifying word, the
judge merged two facts, a line was hallucinated, or a rate-specific note was
read as a site rule. Until now the evidence lived in the run report and the
terminal; this table keeps it, with the model's diagnosis, so a person can
review it and so the same collision is not re-diagnosed on every run.

One action is automatic and recorded here: `rename_new`, which gives the new
statement its own subject and undoes the merge that folded it in (measured on
the last three runs' collisions, experiments.md 2026-09-05 §16-§17: the model
chooses and names this one well; every other action it proposed was either
never chosen or badly named, so those stay manual). `action = 'none'` leaves
the case `open`.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "031_conflict_cases"
down_revision: Union[str, None] = "030_subject_category_split"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "conflict_cases",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("campsite_id", sa.BigInteger(), sa.ForeignKey("campsites.id", ondelete="CASCADE"), nullable=False),
        sa.Column("subject_id", sa.BigInteger(), sa.ForeignKey("subject_vectors.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("kept_term", sa.Text()),
        sa.Column("kept_section", sa.Text()),
        sa.Column("kept_polarity", sa.Boolean()),
        sa.Column("kept_qualifier", sa.Numeric()),
        sa.Column("kept_qualifier_unit", sa.SmallInteger()),
        sa.Column("kept_evidence", sa.Text()),
        sa.Column("kept_resolution", sa.Text()),
        sa.Column("new_term", sa.Text()),
        sa.Column("new_section", sa.Text()),
        sa.Column("new_polarity", sa.Boolean()),
        sa.Column("new_qualifier", sa.Numeric()),
        sa.Column("new_qualifier_unit", sa.SmallInteger()),
        sa.Column("new_evidence", sa.Text()),
        sa.Column("new_resolution", sa.Text()),
        sa.Column("cause", sa.Text()),
        sa.Column("which_is_right", sa.Text()),
        sa.Column("explanation", sa.Text()),
        sa.Column("rationale", sa.Text()),
        sa.Column("confidence", sa.Float()),
        sa.Column("action", sa.Text(), nullable=False, server_default="none"),
        sa.Column("new_name", sa.Text()),
        sa.Column("applied", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("applied_subject_id", sa.BigInteger(), sa.ForeignKey("subject_vectors.id", ondelete="SET NULL")),
        sa.Column("status", sa.Text(), nullable=False, server_default="open"),
        sa.Column("review_note", sa.Text()),
        sa.CheckConstraint("action IN ('none', 'rename_new')", name="conflict_cases_action_check"),
        sa.CheckConstraint("status IN ('open', 'applied', 'reviewed')", name="conflict_cases_status_check"),
    )
    op.create_index("conflict_cases_status_idx", "conflict_cases", ["status"])
    op.create_index("conflict_cases_run_at_idx", "conflict_cases", ["run_at"])


def downgrade() -> None:
    op.drop_index("conflict_cases_run_at_idx", table_name="conflict_cases")
    op.drop_index("conflict_cases_status_idx", table_name="conflict_cases")
    op.drop_table("conflict_cases")
