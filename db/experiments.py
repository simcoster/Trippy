"""Clone production tables into `experiments`, optionally empty some of them.

No foreign key may cross into `public`. `clone_tables` copies DDL with
`LIKE ... INCLUDING ALL` and replays FKs so both ends live in `experiments`.
`copy_public` then fills those tables from `public.*`. Views are recreated
after the copy so search that reads a view still works.

Alembic's version table stays in `public` — this schema is not migrated.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import psycopg

SCHEMA = "experiments"
SKIP_TABLES = frozenset({"alembic_version"})
_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

SEARCH_PATH = "-csearch_path=experiments,extensions"

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

SERIALS_SQL = """
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_schema = 'experiments'
  AND table_name = ANY(%(tables)s)
  AND column_default LIKE 'nextval(%%'
"""

INDEXES_SQL = """
SELECT schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE schemaname = ANY(ARRAY['public', 'experiments'])
  AND tablename = ANY(%(tables)s)
"""


def table_name(name: str) -> str:
    if not _IDENT_RE.fullmatch(name or ""):
        raise ValueError(f"not a plain table identifier: {name!r}")
    return name


def index_shape(indexdef: str) -> tuple[bool, str]:
    """What an index *is*, with its name and schema removed: uniqueness and key."""
    head, _, key = indexdef.partition(" USING ")
    return head.startswith("CREATE UNIQUE"), key


def public_base_tables(cur) -> tuple[str, ...]:
    """Ordinary `public` tables, minus Alembic's bookkeeping."""
    cur.execute(
        """
        SELECT c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relkind = 'r'
        ORDER BY c.relname
        """
    )
    return tuple(
        name for (name,) in cur.fetchall() if name not in SKIP_TABLES
    )


def clone_tables(cur, tables: Sequence[str]) -> None:
    """Rebuild `tables` in `experiments` from production's DDL, empty.

    Dropped and recreated rather than emptied, so a migration that changes a
    column or a constraint is picked up on the next run without anyone
    remembering to do anything.
    """
    names = tuple(table_name(n) for n in tables)
    cur.execute("CREATE SCHEMA IF NOT EXISTS experiments")
    for name in reversed(names):
        cur.execute(f"DROP TABLE IF EXISTS experiments.{name} CASCADE")
    for name in names:
        cur.execute(
            f"CREATE TABLE experiments.{name} (LIKE public.{name} INCLUDING ALL)"
        )
    cur.execute(INDEXES_SQL, {"tables": list(names)})
    produced: dict[tuple[str, bool, str], str] = {}
    cloned: dict[tuple[str, bool, str], str] = {}
    for schema, table, name, definition in cur.fetchall():
        target = produced if schema == "public" else cloned
        target[(table, *index_shape(definition))] = name
    for shape, name in produced.items():
        here = cloned.get(shape)
        if here and here != name:
            cur.execute(f"ALTER INDEX experiments.{here} RENAME TO {name}")

    cur.execute(SERIALS_SQL, {"tables": list(names)})
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

    cur.execute("SET LOCAL search_path TO public")
    cur.execute(FOREIGN_KEYS_SQL, {"tables": list(names)})
    keys = cur.fetchall()
    cur.execute("SET LOCAL search_path TO experiments")
    for table, definition in keys:
        assert "public." not in definition, (
            f"{table}: replaying {definition!r} would point a test key at "
            "production; see .cursor/rules/no-test-data-in-prod.mdc"
        )
        cur.execute(f"ALTER TABLE experiments.{table} ADD {definition}")


def _insert_copy(cur, name: str) -> int:
    """Copy every row. Identity tables need `OVERRIDING SYSTEM VALUE`."""
    sql = f"INSERT INTO experiments.{name} SELECT * FROM public.{name}"
    cur.execute("SAVEPOINT copy_table")
    try:
        cur.execute(sql)
    except psycopg.errors.IntegrityError:
        cur.execute("ROLLBACK TO SAVEPOINT copy_table")
        cur.execute("RELEASE SAVEPOINT copy_table")
        raise
    except psycopg.Error:
        cur.execute("ROLLBACK TO SAVEPOINT copy_table")
        cur.execute(
            f"INSERT INTO experiments.{name} OVERRIDING SYSTEM VALUE "
            f"SELECT * FROM public.{name}"
        )
    cur.execute("RELEASE SAVEPOINT copy_table")
    return cur.rowcount


def _bump_sequences(cur, tables: Sequence[str]) -> None:
    cur.execute(SERIALS_SQL, {"tables": list(tables)})
    for table, column in cur.fetchall():
        cur.execute(
            f"SELECT setval("
            f"pg_get_serial_sequence('experiments.{table}', %s), "
            f"COALESCE((SELECT max({column}) FROM experiments.{table}), 1)"
            f")",
            (column,),
        )


def recreate_views(cur) -> None:
    """Views whose names live in `public`, built against `experiments` tables."""
    cur.execute(
        """
        SELECT c.relname, pg_get_viewdef(c.oid, true)
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'v'
        ORDER BY c.relname
        """
    )
    views = [(table_name(name), definition) for name, definition in cur.fetchall()]
    for name, _ in views:
        cur.execute(f"DROP VIEW IF EXISTS experiments.{name} CASCADE")
    cur.execute("SET LOCAL search_path TO experiments, extensions")
    remaining = list(views)
    for _ in range(len(remaining) + 1):
        if not remaining:
            return
        failed: list[tuple[str, str]] = []
        for name, definition in remaining:
            cur.execute("SAVEPOINT view_try")
            try:
                cur.execute(f"CREATE OR REPLACE VIEW {name} AS {definition}")
            except psycopg.Error:
                cur.execute("ROLLBACK TO SAVEPOINT view_try")
                failed.append((name, definition))
            else:
                cur.execute("RELEASE SAVEPOINT view_try")
        if len(failed) == len(remaining):
            raise RuntimeError(
                "could not create experiments views: "
                + ", ".join(name for name, _ in failed)
            )
        remaining = failed


def empty_tables(cur, tables: Sequence[str]) -> None:
    """TRUNCATE … RESTART IDENTITY CASCADE. Related tables in the schema go too."""
    names = [table_name(n) for n in tables]
    if not names:
        return
    joined = ", ".join(f"experiments.{n}" for n in names)
    cur.execute(f"TRUNCATE {joined} RESTART IDENTITY CASCADE")


def copy_public(
    cur, *, empty: Sequence[str] = (), skip: Sequence[str] = ()
) -> tuple[str, ...]:
    """Rebuild `experiments` as a data copy of `public`, then optionally empty.

    Returns the table names that were cloned. Views are recreated after the
    copy. `empty` is TRUNCATE CASCADE after the fill — use it to start a
    scrape from a full catalog with blank rules, for example.

    `skip` tables are still cloned (empty) so FKs onto them survive the
    DROP CASCADE of parents, but rows are not copied from `public`. The
    planner eval skips `availability` this way and reads
    `availability_frozen` instead.
    """
    tables = public_base_tables(cur)
    skip_set = {table_name(n) for n in skip}
    unknown = skip_set - set(tables)
    if unknown:
        raise ValueError(f"skip names are not public tables: {sorted(unknown)}")
    clone_tables(cur, tables)
    # Alphabetical order is not FK order (accommodation_types before
    # campsites). Replica skips the checks; both ends are filled in this loop.
    cur.execute("SET LOCAL session_replication_role = replica")
    for name in tables:
        if name in skip_set:
            print(f"    skipped {name}: not copied from public")
            continue
        _insert_copy(cur, name)
        cur.execute(f"SELECT count(*) FROM experiments.{name}")
        n = cur.fetchone()[0]
        print(f"    copied {name}: {n} row(s)")
    cur.execute("SET LOCAL session_replication_role = origin")
    _bump_sequences(cur, tables)
    recreate_views(cur)
    if empty:
        empty_tables(cur, empty)
        print(f"    emptied: {', '.join(empty)}")
    return tables


def freeze_availability(cur) -> int:
    """Snapshot `public.availability` into `experiments.availability_frozen`.

    Live scrapes keep writing `availability`. The planner benchmark reads
    this table so occupancy does not move under ingest/retrieve changes.
    No FK into `public`.
    """
    cur.execute("CREATE SCHEMA IF NOT EXISTS experiments")
    cur.execute("DROP TABLE IF EXISTS experiments.availability_frozen")
    cur.execute(
        "CREATE TABLE experiments.availability_frozen "
        "(LIKE public.availability INCLUDING DEFAULTS INCLUDING IDENTITY)"
    )
    cur.execute(
        "INSERT INTO experiments.availability_frozen "
        "OVERRIDING SYSTEM VALUE SELECT * FROM public.availability"
    )
    n = cur.rowcount
    cur.execute(
        "CREATE INDEX availability_frozen_site_dates_idx "
        "ON experiments.availability_frozen (site_id, start_date, end_date)"
    )
    return n
