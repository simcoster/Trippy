"""Clear claims and reset reviews.is_relevant. Keeps review rows."""

from __future__ import annotations

import os
import sys

import psycopg
from dotenv import load_dotenv

from db.connect import connect

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


CLEAR_CLAIMS_SQL = "DELETE FROM claims"

RESET_RELEVANT_SQL = "UPDATE reviews SET is_relevant = NULL"


def clear_claims(conn) -> tuple[int, int]:
    """Delete every claim and null `reviews.is_relevant`. Returns (claims, reviews)."""
    with conn.cursor() as cur:
        cur.execute(CLEAR_CLAIMS_SQL)
        claims_deleted = cur.rowcount
        cur.execute(RESET_RELEVANT_SQL)
        reviews_reset = cur.rowcount
    return claims_deleted, reviews_reset


def main() -> None:
    _log("clear_claims: start")
    url = database_url()
    host_part = url.split("@")[-1] if "@" in url else url
    _log(f"Connecting to Postgres at {host_part} ...")
    try:
        with connect(url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM claims) AS claims,
                      (SELECT COUNT(*) FROM reviews) AS reviews,
                      (SELECT COUNT(*) FROM reviews
                       WHERE is_relevant IS NOT NULL) AS classified
                    """
                )
                before_claims, before_reviews, before_classified = cur.fetchone()
            _log(
                f"Clearing claims and is_relevant "
                f"(before claims={before_claims} reviews={before_reviews} "
                f"classified={before_classified})."
            )
            claims_deleted, reviews_reset = clear_claims(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM claims) AS claims,
                      (SELECT COUNT(*) FROM reviews) AS reviews,
                      (SELECT COUNT(*) FROM reviews
                       WHERE is_relevant IS NOT NULL) AS classified
                    """
                )
                claims, reviews, classified = cur.fetchone()
            conn.commit()
    except psycopg.OperationalError as exc:
        print(f"Postgres connection failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    _log(
        f"Deleted {claims_deleted} claim(s), reset is_relevant on "
        f"{reviews_reset} review(s)."
    )
    _log(f"claims={claims}  reviews={reviews}  classified={classified}")


if __name__ == "__main__":
    main()
