"""info_website_names; list_prices and types point at it

Revision ID: 013_info_website_names
Revises: 012_drop_availability_price
Create Date: 2026-08-31
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "013_info_website_names"
down_revision: Union[str, Sequence[str], None] = "012_drop_availability_price"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "info_website_names",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("site_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["site_id"],
            ["campsites.id"],
            name="info_website_names_site_id_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "site_id",
            "name",
            name="info_website_names_site_id_name_key",
        ),
    )
    op.create_index(
        "info_website_names_site_idx", "info_website_names", ["site_id"]
    )
    op.add_column(
        "list_prices",
        sa.Column("info_website_name_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "accommodation_types",
        sa.Column("info_website_name_id", sa.BigInteger(), nullable=True),
    )
    op.execute(
        """
        INSERT INTO info_website_names (site_id, name)
        SELECT DISTINCT at.hotel_id, at.name
        FROM list_prices lp
        JOIN accommodation_types at ON at.id = lp.accommodation_type_id
        ON CONFLICT (site_id, name) DO NOTHING
        """
    )
    op.execute(
        """
        UPDATE list_prices lp
        SET info_website_name_id = iwn.id
        FROM accommodation_types at
        JOIN info_website_names iwn
          ON iwn.site_id = at.hotel_id AND iwn.name = at.name
        WHERE lp.accommodation_type_id = at.id
        """
    )
    op.execute(
        """
        UPDATE accommodation_types at
        SET info_website_name_id = iwn.id
        FROM info_website_names iwn
        WHERE iwn.site_id = at.hotel_id AND iwn.name = at.name
        """
    )
    op.alter_column(
        "list_prices",
        "info_website_name_id",
        existing_type=sa.BigInteger(),
        nullable=False,
    )
    op.create_foreign_key(
        "list_prices_info_website_name_id_fkey",
        "list_prices",
        "info_website_names",
        ["info_website_name_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "accommodation_types_info_website_name_id_fkey",
        "accommodation_types",
        "info_website_names",
        ["info_website_name_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.drop_constraint(
        "list_prices_unique_rate", "list_prices", type_="unique"
    )
    op.drop_constraint(
        "list_prices_accommodation_type_id_fkey",
        "list_prices",
        type_="foreignkey",
    )
    op.drop_column("list_prices", "accommodation_type_id")
    op.create_unique_constraint(
        "list_prices_unique_rate",
        "list_prices",
        [
            "info_website_name_id",
            "guest_type",
            "rate_period",
            "rate_class",
        ],
    )


def downgrade() -> None:
    op.add_column(
        "list_prices",
        sa.Column("accommodation_type_id", sa.BigInteger(), nullable=True),
    )
    op.execute(
        """
        UPDATE list_prices lp
        SET accommodation_type_id = at.id
        FROM accommodation_types at
        WHERE at.info_website_name_id = lp.info_website_name_id
        """
    )
    op.drop_constraint(
        "list_prices_unique_rate", "list_prices", type_="unique"
    )
    op.drop_constraint(
        "list_prices_info_website_name_id_fkey",
        "list_prices",
        type_="foreignkey",
    )
    op.drop_constraint(
        "accommodation_types_info_website_name_id_fkey",
        "accommodation_types",
        type_="foreignkey",
    )
    op.drop_column("list_prices", "info_website_name_id")
    op.drop_column("accommodation_types", "info_website_name_id")
    op.alter_column(
        "list_prices",
        "accommodation_type_id",
        existing_type=sa.BigInteger(),
        nullable=False,
    )
    op.create_foreign_key(
        "list_prices_accommodation_type_id_fkey",
        "list_prices",
        "accommodation_types",
        ["accommodation_type_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_unique_constraint(
        "list_prices_unique_rate",
        "list_prices",
        [
            "site_id",
            "accommodation_type_id",
            "guest_type",
            "rate_period",
            "rate_class",
        ],
    )
    op.drop_index("info_website_names_site_idx", table_name="info_website_names")
    op.drop_table("info_website_names")
