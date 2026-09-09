"""Copy `public` into `experiments`, and optionally empty some of the copies.

Production is not written. After this, `TRIPPY_SCHEMA=experiments` makes
scrapes, search and the planner use the copy (see `db.connect`).

    uv run python scripts/setup_experiments.py copy
    uv run python scripts/setup_experiments.py copy --skip availability
    uv run python scripts/setup_experiments.py copy --empty campsite_rules,subject_vectors,conflict_cases
    uv run python scripts/setup_experiments.py empty campsite_rules
    uv run python scripts/setup_experiments.py freeze-availability
    uv run python scripts/setup_experiments.py status

    just setup-experiments copy
    just on-experiments scrape-info -- --site 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# `uv run python scripts/…` puts this file's directory on sys.path, not the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.connect import SCHEMA_ENV, connect, database_url
from db.experiments import (
    SEARCH_PATH,
    copy_public,
    empty_tables,
    freeze_availability,
    public_base_tables,
    table_name,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()


def _names(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(table_name(part.strip()) for part in raw.split(",") if part.strip())


def cmd_copy(empty: tuple[str, ...], skip: tuple[str, ...] = ()) -> None:
    skipped = f", skip {', '.join(skip)}" if skip else ""
    print(f"cloning public → experiments (data included{skipped})")
    with connect(database_url(), options=SEARCH_PATH) as conn:
        with conn.cursor() as cur:
            copy_public(cur, empty=empty, skip=skip)
        conn.commit()
    print("done. run scrapes with TRIPPY_SCHEMA=experiments (just on-experiments …)")


def cmd_empty(tables: tuple[str, ...]) -> None:
    if not tables:
        raise SystemExit("empty: name at least one table")
    with connect(database_url(), options=SEARCH_PATH) as conn:
        with conn.cursor() as cur:
            empty_tables(cur, tables)
        conn.commit()
    print(f"emptied: {', '.join(tables)}")


def cmd_freeze_availability() -> None:
    print("snapshot public.availability → experiments.availability_frozen")
    with connect(database_url(), options=SEARCH_PATH) as conn:
        with conn.cursor() as cur:
            n = freeze_availability(cur)
        conn.commit()
    print(f"froze {n} row(s)")


def cmd_status() -> None:
    schema = (os.environ.get(SCHEMA_ENV) or "").strip() or "(unset — production)"
    print(f"{SCHEMA_ENV}={schema}")
    with connect(database_url(), options=SEARCH_PATH) as conn:
        with conn.cursor() as cur:
            tables = public_base_tables(cur)
            print(f"experiments tables: {len(tables)}")
            for name in tables:
                cur.execute(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'experiments' AND table_name = %s"
                    ")",
                    (name,),
                )
                if not cur.fetchone()[0]:
                    print(f"    {name}: (missing)")
                    continue
                cur.execute(f"SELECT count(*) FROM experiments.{name}")
                print(f"    {name}: {cur.fetchone()[0]} row(s)")
            cur.execute(
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'experiments' "
                "AND table_name = 'availability_frozen'"
                ")"
            )
            if cur.fetchone()[0]:
                cur.execute("SELECT count(*) FROM experiments.availability_frozen")
                print(f"    availability_frozen: {cur.fetchone()[0]} row(s)")
            else:
                print("    availability_frozen: (missing)")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["copy", *argv]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    copy = sub.add_parser("copy", help="Rebuild experiments as a copy of public")
    copy.add_argument(
        "--empty",
        default="",
        help="TRUNCATE CASCADE these tables after the copy (comma-separated)",
    )
    copy.add_argument(
        "--skip",
        default="",
        help="Clone these tables empty; do not copy rows from public",
    )
    empty = sub.add_parser("empty", help="TRUNCATE CASCADE named experiments tables")
    empty.add_argument("tables", help="comma-separated table names")
    sub.add_parser("status", help="Row counts in experiments")
    sub.add_parser(
        "freeze-availability",
        help="Copy public.availability into experiments.availability_frozen",
    )
    args = parser.parse_args(argv)
    if args.cmd == "copy":
        cmd_copy(_names(args.empty), skip=_names(args.skip))
    elif args.cmd == "empty":
        cmd_empty(_names(args.tables))
    elif args.cmd == "freeze-availability":
        cmd_freeze_availability()
    else:
        cmd_status()


if __name__ == "__main__":
    main()
