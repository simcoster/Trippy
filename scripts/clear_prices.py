"""Clear published rate cards. Inverse of `just scrape-prices`.

That job writes `list_prices` and `info_website_names`. Accommodation types
point at a name with ON DELETE SET NULL, so the names can go while types stay;
the type just forgets which rate-card product it matched. Availability and
types are `just clear-availability --types`.

`list_prices` is a leaf, so it can TRUNCATE. `info_website_names` cannot:
`accommodation_types` still references it even when empty, and naming that
table here would empty the types. DELETE the names instead.

  uv run python scripts/clear_prices.py
"""

from __future__ import annotations

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
  (SELECT COUNT(*) FROM list_prices),
  (SELECT COUNT(*) FROM info_website_names)
"""


def main() -> None:
    _log("clear_prices: start")
    url = database_url()
    host_part = url.split("@")[-1] if "@" in url else url
    _log(f"Connecting to Postgres at {host_part} ...")
    _log("Deleting list_prices and info_website_names.")

    try:
        with psycopg.connect(url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(COUNTS)
                before_prices, before_names = cur.fetchone()
                _log(
                    f"before: list_prices={before_prices} "
                    f"info_website_names={before_names}"
                )

                cur.execute("TRUNCATE TABLE list_prices RESTART IDENTITY")
                cur.execute("DELETE FROM info_website_names")
                names_deleted = cur.rowcount

                cur.execute(COUNTS)
                prices, names = cur.fetchone()
            conn.commit()
    except psycopg.OperationalError as exc:
        print(f"Postgres connection failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    _log(
        f"Removed {before_prices} list price(s) and {names_deleted} name(s)."
    )
    _log(f"after:  list_prices={prices} info_website_names={names}")
    if before_prices or names_deleted:
        _log("Rebuild with: just scrape-prices")


if __name__ == "__main__":
    main()
