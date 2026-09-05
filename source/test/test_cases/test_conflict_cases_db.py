"""Migration 031 and the apply step against the live database. Every test rolls back."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import psycopg
import pytest
from dotenv import load_dotenv

from source.scraper.rules_ingest.db import DroppedRule, ResolvedRule
from source.scraper.rules_ingest.resolve_conflicts import (
    ConflictResolution,
    apply_rename_new,
    file_conflict_case,
    subject_facts,
)

load_dotenv()


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL required")
    return url.replace("@db:", "@localhost:")


@pytest.fixture
def conn():
    try:
        connection = psycopg.connect(_db_url(), connect_timeout=5)
    except psycopg.OperationalError as exc:
        pytest.skip(f"database unavailable: {exc}")
    with connection:
        yield connection
        connection.rollback()


@pytest.fixture
def scratch(conn):
    """A campsite, an old subject holding a merged alias, and its kept row."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM campsites ORDER BY id LIMIT 1")
        campsite_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO subject_vectors (name, category, aliases) VALUES (%s, 3, %s) RETURNING id",
            ("t_group_min_occupancy", ["t_group_min_occupancy", "t_family_group_min_occupancy"]),
        )
        subject_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO campsite_rules (campsite_id, subject_id, qualifier, qualifier_unit) VALUES (%s, %s, 80, 1)",
            (campsite_id, subject_id),
        )
    drop = DroppedRule(
        "CONFLICTING", campsite_id,
        kept=ResolvedRule(subject_id, None, Decimal("80"), 1, "מעל 80", None, None, "t_group_min_occupancy", "מערכת הזמנות"),
        dropped=ResolvedRule(subject_id, None, Decimal("30"), 1, "30-80", None, None, "t_family_group_min_occupancy", "מערכת הזמנות"),
    )
    return campsite_id, subject_id, drop


def test_the_table_exists_with_its_checks(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM information_schema.columns WHERE table_name = 'conflict_cases'")
        assert cur.fetchone()[0] >= 30
        cur.execute("SELECT id FROM campsites LIMIT 1")
        campsite_id = cur.fetchone()[0]
        cur.execute("SELECT id FROM subject_vectors LIMIT 1")
        subject_id = cur.fetchone()[0]
        with pytest.raises(psycopg.errors.CheckViolation):
            cur.execute(
                "INSERT INTO conflict_cases (run_at, campsite_id, subject_id, label, action) VALUES (now(), %s, %s, 'CONFLICTING', 'reassign_kept')",
                (campsite_id, subject_id),
            )


def test_subject_facts_counts_the_rows(conn, scratch):
    _, subject_id, _ = scratch
    facts = subject_facts(conn, subject_id)
    assert facts.name == "t_group_min_occupancy" and facts.category == 3 and facts.rule_count == 1
    assert "t_family_group_min_occupancy" in facts.aliases


def test_apply_undoes_the_merge_end_to_end(conn, scratch):
    campsite_id, old_id, drop = scratch
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts, **kw: [[0.01] * 1536 for _ in texts]
    resolution = ConflictResolution("judge_over_merge", "both", "two facts", "rename_new", "t_family_group_min_occupancy", "r", 0.95)

    new_id = apply_rename_new(conn, drop, resolution, embedder=embedder)
    resolution.applied, resolution.applied_subject_id = True, new_id
    case_id = file_conflict_case(conn, drop, resolution, run_at=datetime(2026, 9, 5, tzinfo=timezone.utc), kept_how="INSERTED", dropped_how="merged")

    with conn.cursor() as cur:
        cur.execute("SELECT aliases FROM subject_vectors WHERE id = %s", (old_id,))
        assert cur.fetchone()[0] == ["t_group_min_occupancy"]  # alias released, canonical kept
        cur.execute("SELECT name, category, aliases, context, embedding IS NOT NULL FROM subject_vectors WHERE id = %s", (new_id,))
        name, category, aliases, context, has_vector = cur.fetchone()
        assert (name, category, aliases, has_vector) == ("t_family_group_min_occupancy", 3, ["t_family_group_min_occupancy"], True)
        assert context == "מערכת הזמנות: 30-80"
        cur.execute("SELECT subject_id, qualifier FROM campsite_rules WHERE campsite_id = %s AND subject_id IN (%s, %s) ORDER BY subject_id", (campsite_id, old_id, new_id))
        assert cur.fetchall() == [(old_id, Decimal("80")), (new_id, Decimal("30"))]
        cur.execute("SELECT status, applied, applied_subject_id, new_name, kept_qualifier, new_qualifier FROM conflict_cases WHERE id = %s", (case_id,))
        assert cur.fetchone() == ("applied", True, new_id, "t_family_group_min_occupancy", Decimal("80"), Decimal("30"))
