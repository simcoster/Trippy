"""Clear the availability snapshot. Nothing else, unless you ask.

This used to be `just clear-data`, which ran
`TRUNCATE accommodation_types RESTART IDENTITY CASCADE`. TRUNCATE CASCADE is
**table-level**, not row-level: it follows the foreign key from
`campsite_rules.accommodation_type_id` and empties the whole rules table, taking
every site-level rule with it — rows that have no accommodation type at all.
Clearing one night's vacancies destroyed every rule for all 18 campsites, three
separate times, and nothing in the name warned about it.

So the default is availability alone. `--types` opts in to accommodation types
as well, and even then the per-unit rules go by a scoped DELETE, leaving
site-level rules untouched:

  uv run python scripts/clear_availability.py            # availability
  uv run python scripts/clear_availability.py --site 2   # ... for one campsite
  uv run python scripts/clear_availability.py --types    # and accommodation types

Types cost LLM calls to rebuild (`just scrape-availability` re-enriches each one
from its booking tooltip), which is the other reason they are not swept away by
default.

`info_website_names` and `list_prices` are never touched here.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# Allow `uv run python scripts/...` to import sibling scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_db import check_db  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

COUNTS = """
SELECT
  (SELECT COUNT(*) FROM availability),
  (SELECT COUNT(*) FROM accommodation_types),
  (SELECT COUNT(*) FROM campsite_rules WHERE accommodation_type_id IS NOT NULL),
  (SELECT COUNT(*) FROM campsite_rules WHERE accommodation_type_id IS NULL)
"""


def _log(msg: str) -> None:
    print(msg, flush=True)


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        url = "postgresql://trippy:trippy@localhost:5432/trippy"
    return url.replace("@db:", "@localhost:")


def _scope(site: int | None, column: str) -> tuple[str, list]:
    """A WHERE clause covering one campsite *and its subcamps*, or everything.

    A split site's booking units are owned by its subcamp rows, so scoping to
    the parent id alone would leave Akhziv's northern tents behind.
    """
    if site is None:
        return "", []
    return (
        f" WHERE {column} IN ("
        "SELECT id FROM campsites WHERE id = %s OR parent_id = %s)",
        [site, site],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--types",
        action="store_true",
        help="also delete accommodation_types and their per-unit rules",
    )
    parser.add_argument(
        "--site",
        type=int,
        default=None,
        help="limit to one campsite id (its subcamps included)",
    )
    args = parser.parse_args()

    _log("clear_availability: start")
    try:
        check_db()
    except RuntimeError as exc:
        print(exc, file=sys.stderr, flush=True)
        sys.exit(1)

    url = database_url()
    # Avoid printing credentials; show host/db only.
    host_part = url.split("@")[-1] if "@" in url else url
    _log(f"Connecting to Postgres at {host_part} ...")

    target = f" for campsite {args.site} and its subcamps" if args.site else ""
    _log(f"Deleting availability{target}.")
    if args.types:
        _log("Accommodation types and their per-unit rules go too (--types).")
    else:
        _log("Accommodation types are kept; pass --types to remove those too.")
    _log("Site-level rules, info_website_names and list_prices are left in place.")

    try:
        with psycopg.connect(url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(COUNTS)
                before = cur.fetchone()
                _log(
                    f"before: availability={before[0]} types={before[1]} "
                    f"per-unit rules={before[2]} site-level rules={before[3]}"
                )

                clause, params = _scope(args.site, "site_id")
                cur.execute(f"DELETE FROM availability{clause}", params)
                deleted = cur.rowcount

                types_deleted = rules_deleted = 0
                if args.types:
                    # Scoped DELETE, never TRUNCATE ... CASCADE: the rules table
                    # references accommodation_types, and CASCADE would empty
                    # all of it. Per-unit rows first — campsite_rules'
                    # accommodation_type_id is ON DELETE CASCADE, but deleting
                    # them here is what keeps the count honest.
                    hotel_clause, hotel_params = _scope(args.site, "hotel_id")
                    cur.execute(
                        "DELETE FROM campsite_rules WHERE accommodation_type_id IN "
                        f"(SELECT id FROM accommodation_types{hotel_clause})",
                        hotel_params,
                    )
                    rules_deleted = cur.rowcount
                    cur.execute(
                        f"DELETE FROM accommodation_types{hotel_clause}",
                        hotel_params,
                    )
                    types_deleted = cur.rowcount

                # DELETE only marks entries dead; the heap and indexes keep
                # their size. Once a table is empty, TRUNCATE writes fresh files
                # for it and every index, and costs nothing.
                #
                # But TRUNCATE refuses while any table references the target,
                # even an empty one, and the fix for that is CASCADE — the exact
                # thing this script exists to avoid. So name every referencing
                # table in the one statement instead, and only ever name a table
                # that is already empty. `accommodation_types` is referenced by
                # both `availability` and `campsite_rules`, so it can only be
                # reset when all three are empty; if per-unit rules remain
                # elsewhere, the files simply keep their size.
                empty = []
                for table in ("availability", "accommodation_types", "campsite_rules"):
                    cur.execute(f"SELECT EXISTS (SELECT 1 FROM {table})")
                    if not cur.fetchone()[0]:
                        empty.append(table)

                resettable: list[str] = []
                if deleted and "availability" in empty:
                    resettable.append("availability")
                if types_deleted and {"accommodation_types", "campsite_rules"} <= set(
                    empty
                ):
                    if "availability" in resettable:
                        resettable += ["accommodation_types", "campsite_rules"]
                elif types_deleted and "accommodation_types" in empty:
                    _log(
                        "accommodation_types is empty but campsite_rules is not; "
                        "leaving its files at size rather than cascading."
                    )
                if resettable:
                    cur.execute(
                        f"TRUNCATE TABLE {', '.join(resettable)} RESTART IDENTITY"
                    )
                    _log(f"Truncated {', '.join(resettable)} to reset the files.")

                cur.execute(COUNTS)
                after = cur.fetchone()
            conn.commit()
    except psycopg.OperationalError as exc:
        print(f"Postgres connection failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    _log(f"Removed {deleted} availability row(s).")
    if args.types:
        _log(f"Removed {types_deleted} type(s) and {rules_deleted} per-unit rule(s).")
    _log(
        f"after:  availability={after[0]} types={after[1]} "
        f"per-unit rules={after[2]} site-level rules={after[3]}"
    )
    if before[3] != after[3]:
        _log("WARNING: site-level rules changed. That should never happen here.")
    if deleted or types_deleted:
        _log("Rebuild with: just scrape-availability")


if __name__ == "__main__":
    main()
