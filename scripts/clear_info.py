"""Clear everything `just scrape-info` writes: rooms, prices, rules and
breadcrumb region claims.

The inverse of the whole info-page pipeline, where `clear_rules.py` is the
inverse of one step of it. It empties, in the order the foreign keys allow:

    availability          every row pins an accommodation type
    campsite_rules        both scopes -- site-level and per-unit
    conflict_cases        each names its subjects by id, so none survives the
                          renumbering a re-ingest brings
    list_prices           rebuilt by scrape-prices
    accommodation_types   rebuilt by scrape-rooms from the lodging panel
    info_website_names    created by scrape-rooms; the row prices attach to
    subject_vectors       the shared dictionary
    claims (no review)    breadcrumb regions; review-split claims stay

`availability` goes too, and it has to: `availability.accommodation_type_id` is
ON DELETE RESTRICT, so every vacancy pins the type it was booked against and
nothing else could be cleared while one remains. Rebuilding it is a separate
job (`just scrape-availability`, a booking-engine sweep) from rebuilding the
rest (`just scrape-info`, an info-page one), so the summary says so.

TRUNCATE, not DELETE, and one statement for all seven. DELETE only marks entries
dead; VACUUM then reclaims that space for reuse inside the file rather than
returning it, and on the HNSW vector indexes over `subject_vectors` it is worse
than bloat -- HNSW is a graph, deletions unlink nodes without rebuilding it, and
VACUUM cannot. Repeated wipe-and-re-ingest, which is exactly how this project is
used, degrades recall. TRUNCATE writes new empty files for every table and index
instead. It takes several tables at once precisely so a referencing group can go
together without CASCADE reaching anything unnamed. Breadcrumb claims are a
DELETE (`review_id IS NULL`) because review-split rows share the table.

  uv run python scripts/clear_info.py
  uv run python scripts/clear_info.py --yes    # no prompt
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg
from dotenv import load_dotenv

from db.connect import connect

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

BREADCRUMB_CLAIMS_KEY = "claims (no review)"
DELETE_BREADCRUMB_CLAIMS_SQL = "DELETE FROM claims WHERE review_id IS NULL"

# In FK order, though TRUNCATE takes them together.
TABLES = (
    "availability",
    "campsite_rules",
    "conflict_cases",
    "list_prices",
    "accommodation_types",
    "info_website_names",
    "subject_vectors",
)


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        url = "postgresql://trippy:trippy@localhost:5432/trippy"
    return url.replace("@db:", "@localhost:")


def counts(cur) -> dict[str, int]:
    out: dict[str, int] = {}
    for table in TABLES:
        cur.execute(f"SELECT count(*) FROM {table}")
        out[table] = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM claims WHERE review_id IS NULL")
    out[BREADCRUMB_CLAIMS_KEY] = cur.fetchone()[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true", help="skip the confirmation prompt"
    )
    args = parser.parse_args()

    url = database_url()
    print(f"clear_info: {url.split('@')[-1] if '@' in url else url}")
    try:
        with connect(url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                before = counts(cur)
            keys = TABLES + (BREADCRUMB_CLAIMS_KEY,)
            for table in keys:
                print(f"  {table:22} {before[table]:>6}")

            total = sum(before[t] for t in keys)
            if not total:
                print("\nAlready empty; nothing to do.")
                return
            if not args.yes:
                answer = input(f"\nDelete {total} row(s)? [y/N] ").strip().lower()
                if answer not in {"y", "yes"}:
                    print("Left alone.")
                    return

            with conn.cursor() as cur:
                cur.execute(
                    f"TRUNCATE TABLE {', '.join(TABLES)} RESTART IDENTITY"
                )
                cur.execute(DELETE_BREADCRUMB_CLAIMS_SQL)
                after = counts(cur)
            conn.commit()
    except psycopg.OperationalError as exc:
        print(f"Postgres connection failed: {exc}", file=sys.stderr)
        sys.exit(1)

    assert all(after[t] == 0 for t in keys), after
    print(f"\nRemoved {total} row(s); the tables and their indexes are reset.")
    print("Rebuild with: just scrape-info, then just scrape-availability")


if __name__ == "__main__":
    main()
