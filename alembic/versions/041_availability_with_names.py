"""availability_with_names — slot plus campsite and unit names

Revision ID: 041_availability_with_names
Revises: 040_site_price_functions
Create Date: 2026-09-19

`availability` is site_id + accommodation_type_id. This view is for
eyeballing a night: Hebrew campsite name and unit name next to the
dates and room_count. Ids stay so you can still join.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "041_availability_with_names"
down_revision: Union[str, Sequence[str], None] = "040_site_price_functions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VIEW_NAME = "availability_with_names"

CREATE_VIEW_SQL = """
CREATE VIEW availability_with_names AS
SELECT
    a.id,
    c.name AS campsite,
    at.name AS accommodation_type,
    a.start_date,
    a.end_date,
    a.room_count,
    a.site_id,
    a.accommodation_type_id,
    a.scraped_at,
    a.updated_at
FROM availability a
JOIN campsites c ON c.id = a.site_id
JOIN accommodation_types at ON at.id = a.accommodation_type_id
"""


def upgrade() -> None:
    op.execute(CREATE_VIEW_SQL)


def downgrade() -> None:
    op.execute(f"DROP VIEW IF EXISTS {VIEW_NAME}")
