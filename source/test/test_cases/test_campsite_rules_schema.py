"""Migrations 023/024 against a live database: the constraints must actually bite.

Every test rolls back, so the dev database is left untouched.
"""

from __future__ import annotations

import os

import psycopg
import pytest
from dotenv import load_dotenv

load_dotenv()


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    assert url, "DATABASE_URL is required"
    return url.replace("@db:", "@localhost:")


@pytest.fixture
def conn():
    with psycopg.connect(_db_url()) as connection:
        yield connection
        connection.rollback()


@pytest.fixture
def scratch(conn):
    """An existing campsite plus a throwaway subject, rolled back afterwards."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM campsites ORDER BY id LIMIT 1")
        row = cur.fetchone()
        if row is None:
            pytest.skip("no campsites in the database")
        cur.execute(
            """
            INSERT INTO subject_vectors (name, category, aliases)
            VALUES ('test_schema_subject', 2, ARRAY['test_schema_subject'])
            RETURNING id
            """
        )
        yield row[0], cur.fetchone()[0], cur


def test_expected_indexes_exist(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexname FROM pg_indexes
            WHERE tablename IN ('subject_vectors', 'campsite_rules')
            """
        )
        names = {r[0] for r in cur.fetchall()}
    assert {
        "subject_vectors_aliases_gin_idx",
        "subject_vectors_embedding_idx",
        "subject_vectors_embedding_amenity_idx",
        "subject_vectors_embedding_rule_idx",
        "campsite_rules_scope_subject_key",
        "campsite_rules_subject_qualifier_idx",
        "campsite_rules_accom_idx",
    } <= names


def test_the_embedding_index_ranks_by_inner_product(conn):
    """Every query in the repo uses `<#>`; a cosine index would never be used."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = %s",
            ("subject_vectors_embedding_idx",),
        )
        definition = cur.fetchone()[0]
    assert "vector_ip_ops" in definition
    assert "hnsw" in definition.lower()


def test_each_category_has_its_own_partial_vector_index(conn):
    """So a rule is never even scanned when searching for an amenity."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE indexname LIKE 'subject_vectors_embedding_%_idx'"
        )
        by_name = dict(cur.fetchall())
    assert "WHERE (category = 1)" in by_name["subject_vectors_embedding_amenity_idx"]
    assert "WHERE (category = 2)" in by_name["subject_vectors_embedding_rule_idx"]
    for definition in by_name.values():
        assert "vector_ip_ops" in definition


def test_one_query_can_run_a_nearest_neighbour_search_per_category(conn):
    """UNION ALL of two ORDER BY/LIMIT branches — one round trip, two probes."""
    with conn.cursor() as cur:
        cur.execute("SELECT embedding FROM subject_vectors WHERE embedding IS NOT NULL LIMIT 1")
        row = cur.fetchone()
        if row is None:
            pytest.skip("no embedded subjects")
        cur.execute(
            """
            (SELECT 1 AS category, name FROM subject_vectors
               WHERE category = 1 AND embedding IS NOT NULL
               ORDER BY embedding <#> %(v)s::vector LIMIT 3)
            UNION ALL
            (SELECT 2, name FROM subject_vectors
               WHERE category = 2 AND embedding IS NOT NULL
               ORDER BY embedding <#> %(v)s::vector LIMIT 3)
            """,
            {"v": row[0]},
        )
        hits = cur.fetchall()
    assert {c for c, _ in hits} <= {1, 2}
    assert len(hits) > 1


def test_every_pre_existing_subject_was_backfilled_with_its_name_as_alias(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM subject_vectors WHERE aliases IS NULL OR aliases[1] <> name"
        )
        assert cur.fetchone()[0] == 0


def test_the_amenity_name_views_followed_the_table_rename(conn):
    """Postgres keys view definitions on OIDs, so no view rebuild was needed."""
    with conn.cursor() as cur:
        for view in (
            "campsites_with_amenity_names",
            "accommodation_types_with_amenity_names",
        ):
            cur.execute("SELECT pg_get_viewdef(%s::regclass)", (view,))
            assert "subject_vectors" in cur.fetchone()[0]


def test_site_level_rows_dedupe_on_reingest(scratch):
    """NULLS NOT DISTINCT: without it every re-ingest would duplicate site rules."""
    campsite_id, subject_id, cur = scratch
    for qualifier in (12, 14):
        cur.execute(
            """
            INSERT INTO campsite_rules
                (campsite_id, accommodation_type_id, subject_id, qualifier, qualifier_unit)
            VALUES (%s, NULL, %s, %s, 2)
            ON CONFLICT (campsite_id, accommodation_type_id, subject_id) DO UPDATE
            SET qualifier = EXCLUDED.qualifier
            """,
            (campsite_id, subject_id, qualifier),
        )
    cur.execute(
        "SELECT count(*), max(qualifier) FROM campsite_rules WHERE subject_id = %s",
        (subject_id,),
    )
    assert cur.fetchone() == (1, 14)


def test_a_unit_scoped_row_coexists_with_the_site_scoped_one(scratch):
    campsite_id, subject_id, cur = scratch
    cur.execute(
        "SELECT id FROM accommodation_types WHERE hotel_id = %s LIMIT 1", (campsite_id,)
    )
    row = cur.fetchone()
    if row is None:
        pytest.skip("no accommodation types for this campsite")
    for accom_id in (None, row[0]):
        cur.execute(
            """
            INSERT INTO campsite_rules (campsite_id, accommodation_type_id, subject_id)
            VALUES (%s, %s, %s)
            """,
            (campsite_id, accom_id, subject_id),
        )
    cur.execute(
        "SELECT count(*) FROM campsite_rules WHERE subject_id = %s", (subject_id,)
    )
    assert cur.fetchone()[0] == 2


def test_the_canonical_alias_check_rejects_a_mismatched_first_alias(conn):
    with conn.cursor() as cur, pytest.raises(psycopg.errors.CheckViolation):
        cur.execute(
            "INSERT INTO subject_vectors (name, category, aliases) "
            "VALUES ('bad_alias_row', 1, ARRAY['something_else'])"
        )


def test_the_category_check_rejects_an_unknown_category(conn):
    with conn.cursor() as cur, pytest.raises(psycopg.errors.CheckViolation):
        cur.execute(
            "INSERT INTO subject_vectors (name, category, aliases) "
            "VALUES ('bad_category_row', 7, ARRAY['bad_category_row'])"
        )


def test_an_alias_lookup_matches_exactly_not_by_prefix(scratch):
    _campsite_id, subject_id, cur = scratch
    cur.execute(
        "UPDATE subject_vectors SET aliases = aliases || ARRAY['test_schema_alias'] "
        "WHERE id = %s",
        (subject_id,),
    )
    cur.execute(
        "SELECT id FROM subject_vectors WHERE aliases @> ARRAY['test_schema_alias']"
    )
    assert cur.fetchone()[0] == subject_id
    cur.execute("SELECT id FROM subject_vectors WHERE aliases @> ARRAY['test_schema']")
    assert cur.fetchone() is None
