"""Clear accommodation_types and availability tables (TRUNCATE … CASCADE).

Does not delete info_website_names or list_prices.
"""

from __future__ import annotations

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


def _log(msg: str) -> None:
    print(msg, flush=True)


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        url = "postgresql://trippy:trippy@localhost:5432/trippy"
    return url.replace("@db:", "@localhost:")


def main() -> None:
    _log("clear_accommodation_availability: start")
    try:
        check_db()
    except RuntimeError as exc:
        print(exc, file=sys.stderr, flush=True)
        sys.exit(1)

    url = database_url()
    # Avoid printing credentials; show host/db only.
    host_part = url.split("@")[-1] if "@" in url else url
    _log(f"Connecting to Postgres at {host_part} ...")
    try:
        with psycopg.connect(url, connect_timeout=10) as conn:
            _log("Truncating accommodation_types (CASCADE to availability).")
            _log("info_website_names and list_prices are left in place.")
            with conn.cursor() as cur:
                cur.execute(
                    "TRUNCATE TABLE accommodation_types RESTART IDENTITY CASCADE"
                )
                _log("Truncate done. Counting rows...")
                cur.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM accommodation_types) AS accommodation_types,
                      (SELECT COUNT(*) FROM availability) AS availability
                    """
                )
                accom, avail = cur.fetchone()
            conn.commit()
    except psycopg.OperationalError as exc:
        print(f"Postgres connection failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    _log("Truncated accommodation_types (cascaded to availability).")
    _log(f"accommodation_types={accom}  availability={avail}")


if __name__ == "__main__":
    main()
