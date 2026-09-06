"""Test tables in the `experiments` schema, cloned from production's own DDL.

See `.cursor/rules/no-test-data-in-prod.mdc`: a test may do whatever it likes
here and must never touch a table in `public`. The connection is opened with
`options="-csearch_path=experiments"`, so an unqualified table name -- in the
test, or in the production code it calls -- resolves here or raises.

The tables are **clones, not hand-written copies**:

    CREATE TABLE experiments.campsites (LIKE public.campsites INCLUDING ALL)

which matters because several of these tests exist to prove that production's
constraints bite -- the `aliases[1] = name` check, the unique index on
`campsites.url`, the `qualifier_unit` bounds. A hand-written copy would test the
copy. `LIKE ... INCLUDING ALL` brings the checks, defaults and indexes across,
and stays in step with every migration for free.

`LIKE` does not copy foreign keys, which is exactly right: none may cross into
`public`. The ones the tests need are replayed from production's own
definitions, both ends inside the schema, by `clone_tables`.

The `experiments_conn` fixture that uses all this lives in `conftest.py`.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

# The fixture is imported by conftest, so this runs before any test asks for a
# connection -- without it every database test skips for a missing DATABASE_URL.
load_dotenv()

# `extensions` holds `vector` and `pg_trgm` (migration 033), so the type and
# its operators resolve here. `public` stays off the path: that is what stops
# an unqualified name in a test -- or in the production code it calls --
# reaching a real table.
SEARCH_PATH = "-csearch_path=experiments,extensions"

# Cloned in this order: a foreign key can only be replayed once both ends exist.
TABLES = (
    "campsites",
    "subject_vectors",
    "info_website_names",
    "accommodation_types",
    "campsite_rules",
    "conflict_cases",
    "list_prices",
)

# Far from any real campsite id, so a query that somehow escaped to `public`
# would match nothing rather than something.
SITE_ID = 900001

# Read with `search_path` set to `public`, so `pg_get_constraintdef` writes the
# referenced table unqualified -- `REFERENCES campsites(id)`, not
# `REFERENCES public.campsites(id)`. Replayed under `search_path=experiments`
# that unqualified name resolves inside the schema, which is the whole point:
# qualified, every replayed key pointed back at production.
FOREIGN_KEYS_SQL = """
SELECT t.relname, pg_get_constraintdef(c.oid)
FROM pg_constraint c
JOIN pg_class t ON t.oid = c.conrelid
JOIN pg_class ft ON ft.oid = c.confrelid
JOIN pg_namespace n ON n.oid = t.relnamespace
WHERE c.contype = 'f'
  AND n.nspname = 'public'
  AND t.relname = ANY(%(tables)s)
  AND ft.relname = ANY(%(tables)s)
"""

# A `serial` column's default is `nextval('public.<table>_id_seq')`, and `LIKE`
# copies it verbatim -- so an insert here would draw its id from production's
# sequence. Each clone gets its own instead.
SERIALS_SQL = """
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_schema = 'experiments'
  AND table_name = ANY(%(tables)s)
  AND column_default LIKE 'nextval(%%'
"""


# `LIKE ... INCLUDING INDEXES` copies the index but not its name -- production's
# `list_prices_unique_rate` arrives as `list_prices_info_website_name_id_..._key`.
# Code that says `ON CONFLICT ON CONSTRAINT list_prices_unique_rate` then fails
# here and nowhere else, so the names are put back.
INDEXES_SQL = """
SELECT schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE schemaname = ANY(ARRAY['public', 'experiments'])
  AND tablename = ANY(%(tables)s)
"""


def index_shape(indexdef: str) -> tuple[bool, str]:
    """What an index *is*, with its name and schema removed: uniqueness and key."""
    head, _, key = indexdef.partition(" USING ")
    return head.startswith("CREATE UNIQUE"), key


def db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL is required")
    return url.replace("@db:", "@localhost:")


def clone_tables(cur, tables: tuple[str, ...] = TABLES) -> None:
    """Rebuild `tables` in `experiments` from production's DDL, empty.

    Dropped and recreated rather than emptied, so a migration that changes a
    column or a constraint is picked up on the next run without anyone
    remembering to do anything.
    """
    cur.execute("CREATE SCHEMA IF NOT EXISTS experiments")
    for name in reversed(tables):
        cur.execute(f"DROP TABLE IF EXISTS experiments.{name} CASCADE")
    for name in tables:
        cur.execute(
            f"CREATE TABLE experiments.{name} (LIKE public.{name} INCLUDING ALL)"
        )
    cur.execute(INDEXES_SQL, {"tables": list(tables)})
    # Keyed by table as well as shape: every table has a `(id)` primary key, and
    # keying on the shape alone renamed one table's index to another's name.
    produced: dict[tuple[str, bool, str], str] = {}
    cloned: dict[tuple[str, bool, str], str] = {}
    for schema, table, name, definition in cur.fetchall():
        target = produced if schema == "public" else cloned
        target[(table, *index_shape(definition))] = name
    for shape, name in produced.items():
        here = cloned.get(shape)
        if here and here != name:
            # Renaming a constraint's index renames the constraint with it.
            cur.execute(f"ALTER INDEX experiments.{here} RENAME TO {name}")

    cur.execute(SERIALS_SQL, {"tables": list(tables)})
    for table, column in cur.fetchall():
        cur.execute(f"CREATE SEQUENCE IF NOT EXISTS experiments.{table}_{column}_seq")
        cur.execute(
            f"ALTER TABLE experiments.{table} ALTER COLUMN {column} "
            f"SET DEFAULT nextval('experiments.{table}_{column}_seq')"
        )
        cur.execute(
            f"ALTER SEQUENCE experiments.{table}_{column}_seq "
            f"OWNED BY experiments.{table}.{column}"
        )

    # `LIKE` leaves the foreign keys behind. Replay production's own definitions
    # so `ON DELETE CASCADE` and `RESTRICT` behave as the tests expect -- both
    # ends are inside `experiments`, so nothing crosses into `public`.
    cur.execute("SET LOCAL search_path TO public")
    cur.execute(FOREIGN_KEYS_SQL, {"tables": list(tables)})
    keys = cur.fetchall()
    # Back before the first `ALTER`: an unqualified `REFERENCES campsites` has
    # to resolve here, not in `public`.
    cur.execute("SET LOCAL search_path TO experiments")
    for table, definition in keys:
        assert "public." not in definition, (
            f"{table}: replaying {definition!r} would point a test key at "
            "production; see .cursor/rules/no-test-data-in-prod.mdc"
        )
        cur.execute(f"ALTER TABLE experiments.{table} ADD {definition}")
