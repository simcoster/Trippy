"""pgvector and pg_trgm move to their own schema, reachable from `experiments`

Revision ID: 033_extensions_schema
Revises: 032_types_from_panel
Create Date: 2026-09-06

An extension is installed once per database, into one schema. `vector` and
`pg_trgm` were in `public`, and the tests open their connection with
`options="-csearch_path=experiments"` -- `public` deliberately absent, so that
an unqualified table name cannot resolve to a production table
(`.cursor/rules/no-test-data-in-prod.mdc`). That also put the `vector` type and
its operators out of reach, so `%(embedding)s::vector` in `subjects/resolve.py`
failed in the test schema and nowhere else.

The type belongs to neither schema. It goes in `extensions`, which both can
see: production connects with `"$user", public, extensions` and the tests with
`experiments, extensions`. Nothing production writes moves, and `public` stays
off the test path.

The alternative was to append `public` to the test search_path, which is the
one thing the rule exists to prevent: a table that had not been cloned would
resolve to the real one and the test would quietly edit production data.

Existing connections keep the search_path they opened with, so anything already
running -- the `api` container especially -- must reconnect before an
unqualified `vector` resolves again.
"""

from alembic import op

revision = "033_extensions_schema"
down_revision = "032_types_from_panel"
branch_labels = None
depends_on = None

EXTENSIONS = ("vector", "pg_trgm")


def database_name() -> str:
    """Quoted, because `ALTER DATABASE` takes an identifier, not a parameter."""
    conn = op.get_bind()
    name = conn.exec_driver_sql("SELECT current_database()").scalar_one()
    return '"%s"' % name.replace('"', '""')


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS extensions")
    for name in EXTENSIONS:
        op.execute(f"ALTER EXTENSION {name} SET SCHEMA extensions")
    # Appended, not replacing: `public` stays first so every unqualified table
    # name in the application resolves exactly as it did before.
    op.execute(
        f'ALTER DATABASE {database_name()} SET search_path TO "$user", public, extensions'
    )


def downgrade() -> None:
    op.execute(f'ALTER DATABASE {database_name()} SET search_path TO "$user", public')
    for name in EXTENSIONS:
        op.execute(f"ALTER EXTENSION {name} SET SCHEMA public")
    op.execute("DROP SCHEMA IF EXISTS extensions")
