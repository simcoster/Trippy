"""Clear reviews and claims tables (TRUNCATE … CASCADE).

Does not delete campsites.
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


def main() -> None:
    _log("clear_reviews_and_claims: start")
    url = database_url()
    host_part = url.split("@")[-1] if "@" in url else url
    _log(f"Connecting to Postgres at {host_part} ...")
    try:
        with psycopg.connect(url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM reviews) AS reviews,
                      (SELECT COUNT(*) FROM claims) AS claims
                    """
                )
                before_reviews, before_claims = cur.fetchone()
                _log(
                    f"Truncating reviews and claims "
                    f"(before reviews={before_reviews} claims={before_claims})."
                )
                cur.execute("TRUNCATE TABLE claims, reviews RESTART IDENTITY CASCADE")
                cur.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM reviews) AS reviews,
                      (SELECT COUNT(*) FROM claims) AS claims
                    """
                )
                reviews, claims = cur.fetchone()
            conn.commit()
    except psycopg.OperationalError as exc:
        print(f"Postgres connection failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    _log("Truncated reviews and claims.")
    _log(f"reviews={reviews}  claims={claims}")


if __name__ == "__main__":
    main()
