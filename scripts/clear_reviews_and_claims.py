"""Clear Google review rows and the claims split from them.

Does not delete campsites. Breadcrumb region claims (`review_id` NULL)
stay — they are an info-page ingest, rebuilt by `just scrape-breadcrumbs`.
"""

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


def main() -> None:
    _log("clear_reviews_and_claims: start")
    url = database_url()
    host_part = url.split("@")[-1] if "@" in url else url
    _log(f"Connecting to Postgres at {host_part} ...")
    try:
        with connect(url, connect_timeout=10) as conn:
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
                    f"Deleting reviews (CASCADE drops review-split claims; "
                    f"before reviews={before_reviews} claims={before_claims})."
                )
                cur.execute("DELETE FROM reviews")
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

    _log("Deleted reviews (review-split claims cascaded).")
    _log(f"reviews={reviews}  claims={claims}")


if __name__ == "__main__":
    main()
