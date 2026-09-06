"""Insert the rows CI tests hang scratch data off.

test_conflict_cases_db and test_campsite_rules_schema select the first
campsite; test_the_table_exists_with_its_checks also fetchone()s
subject_vectors. An empty table is a TypeError or a skip.
"""

from __future__ import annotations

import os

import psycopg


def main() -> None:
    url = os.environ["DATABASE_URL"]
    with psycopg.connect(url) as conn:
        conn.execute(
            "INSERT INTO campsites (name, url) VALUES (%s, %s)",
            ("ci-seed", "https://ci.invalid/"),
        )
        conn.execute(
            "INSERT INTO subject_vectors (name, category, aliases) VALUES (%s, %s, %s)",
            ("ci_seed_subject", 3, ["ci_seed_subject"]),
        )


if __name__ == "__main__":
    main()
