"""Clear campsite_rules, without touching the vocabulary that gives it meaning.

The inverse of `just ingest-rules`, so by default it deletes exactly what that
job writes: **site-level rows only** (`accommodation_type_id IS NULL`). Per-unit
rows come from the availability scrape's amenity enrichment and can only be
rebuilt by re-running it, so they are kept unless you ask for them.

Scoped `DELETE`, never `TRUNCATE ... CASCADE`. That distinction is the reason
this script exists: TRUNCATE CASCADE is **table-level**, not row-level, so
`TRUNCATE accommodation_types CASCADE` — what `just clear-data` used to run —
empties the whole of `campsite_rules`, site-level rules included, even though
those rows have no accommodation type at all. Losing every rule for all 18
campsites while clearing availability is not something you should have to
predict; `just clear-availability` no longer does.

But TRUNCATE on the *right* table is worth having, because DELETE does not free
anything: it marks entries dead, and VACUUM reclaims that space for reuse inside
the file rather than returning it. On the HNSW vector indexes over
`subject_vectors` it is worse than bloat — HNSW is a graph, deletions unlink
nodes without rebuilding it, and VACUUM cannot. Repeated wipe-and-re-ingest,
which is exactly how this project is used, degrades recall. TRUNCATE writes new
empty files for the table and every index, so it resets them for free.

So: TRUNCATE whenever the whole table is going anyway — `--subjects`, or a
delete that happens to empty it — and a scoped DELETE otherwise, with a note
that a partial clear leaves the indexes at full size.

`subject_vectors` is left alone by default. It is the shared dictionary: its
rows carry embeddings that cost LLM calls to rebuild, and its aliases are what
make a re-ingest converge on the same subjects instead of forking new ones.
`--subjects` clears it too, and implies `--all`, because `campsite_rules.
subject_id` is ON DELETE RESTRICT — a subject cannot go while any rule cites it.

  uv run python scripts/clear_rules.py                # site-level rules
  uv run python scripts/clear_rules.py --site 2       # ... for one campsite
  uv run python scripts/clear_rules.py --all          # per-unit rows as well
  uv run python scripts/clear_rules.py --subjects     # and the vocabulary
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()


def _log(msg: str) -> None:
    print(msg, flush=True)


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        url = "postgresql://trippy:trippy@localhost:5432/trippy"
    return url.replace("@db:", "@localhost:")


COUNTS = """
SELECT
  (SELECT COUNT(*) FROM campsite_rules WHERE accommodation_type_id IS NULL),
  (SELECT COUNT(*) FROM campsite_rules WHERE accommodation_type_id IS NOT NULL),
  (SELECT COUNT(*) FROM subject_vectors)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="also delete per-unit rows (only the availability scrape can rebuild them)",
    )
    parser.add_argument(
        "--subjects",
        action="store_true",
        help="also clear subject_vectors; implies --all",
    )
    parser.add_argument(
        "--site",
        type=int,
        default=None,
        help="limit to one campsite id (not allowed with --subjects)",
    )
    args = parser.parse_args()

    if args.subjects and args.site is not None:
        parser.error(
            "--subjects clears the whole shared dictionary, so it cannot be "
            "scoped to one site"
        )
    include_units = args.all or args.subjects

    _log("clear_rules: start")
    url = database_url()
    host_part = url.split("@")[-1] if "@" in url else url
    _log(f"Connecting to Postgres at {host_part} ...")

    where = [] if include_units else ["accommodation_type_id IS NULL"]
    params: list = []
    if args.site is not None:
        where.append("campsite_id = %s")
        params.append(args.site)
    clause = f" WHERE {' AND '.join(where)}" if where else ""

    scope = "all rules" if include_units else "site-level rules"
    target = f" for campsite {args.site}" if args.site is not None else ""
    _log(f"Deleting {scope}{target}.")
    if not include_units:
        _log("Per-unit rows are kept; pass --all to remove those too.")

    try:
        with psycopg.connect(url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(COUNTS)
                before_site, before_unit, before_subjects = cur.fetchone()
                _log(
                    f"before: site-level={before_site} per-unit={before_unit} "
                    f"subjects={before_subjects}"
                )

                subjects_deleted = 0
                if args.subjects:
                    # One statement for both, because campsite_rules references
                    # subject_vectors with ON DELETE RESTRICT: TRUNCATE accepts
                    # several tables precisely so a referencing pair can go
                    # together without CASCADE reaching anything unnamed.
                    _log("Truncating campsite_rules and subject_vectors.")
                    cur.execute(
                        "TRUNCATE TABLE campsite_rules, subject_vectors "
                        "RESTART IDENTITY"
                    )
                    deleted, subjects_deleted = (
                        before_site + before_unit,
                        before_subjects,
                    )
                else:
                    cur.execute(f"DELETE FROM campsite_rules{clause}", params)
                    deleted = cur.rowcount
                    # DELETE leaves the heap and every index at full size: the
                    # entries are only marked dead, and VACUUM reclaims that
                    # space for reuse rather than returning it. On the HNSW
                    # vector indexes that also leaves the graph un-rebuilt,
                    # which degrades recall across repeated wipe-and-re-ingest
                    # cycles. Once the table is empty an extra TRUNCATE is
                    # nearly free and resets the files, so take it.
                    cur.execute("SELECT EXISTS (SELECT 1 FROM campsite_rules)")
                    if deleted and not cur.fetchone()[0]:
                        _log("Table is now empty; truncating to reset the files.")
                        cur.execute(
                            "TRUNCATE TABLE campsite_rules RESTART IDENTITY"
                        )

                cur.execute(COUNTS)
                site, unit, subjects = cur.fetchone()
            conn.commit()
    except psycopg.errors.ForeignKeyViolation as exc:
        print(
            f"Refused: a rule still cites a subject ({exc}). "
            "Use --subjects, which clears the rules first.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    except psycopg.OperationalError as exc:
        print(f"Postgres connection failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    tail = f" and {subjects_deleted} subject(s)." if args.subjects else "."
    _log(f"Removed {deleted} rule(s){tail}")
    _log(f"after:  site-level={site} per-unit={unit} subjects={subjects}")
    if deleted and (site or unit):
        # A partial delete cannot reset the files, and the HNSW indexes on
        # subject_vectors are the ones that degrade rather than merely bloat.
        _log(
            "Rows remain, so the index files keep their size. "
            "REINDEX TABLE subject_vectors if you do this often."
        )
    if deleted:
        _log("Rebuild with: just ingest-rules")


if __name__ == "__main__":
    main()
