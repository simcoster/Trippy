"""Lifetime public-demo cap: hash the visitor, count in experiments only."""

from __future__ import annotations

import psycopg
import pytest

from source.demo_quota import (
    PEPPER_ENV,
    VISITOR_IP_ENV,
    claim_query,
    public_visitor_hash,
    quota_remaining,
    remaining_caption,
    visitor_hash,
    visitor_ip,
)
from source.test.test_cases.experiments_schema import SEARCH_PATH, db_url

_CAP = 5
_PEPPER = "test-pepper"
_IP = "203.0.113.8"


def test_visitor_ip_reads_cloudflare_header():
    assert visitor_ip({"CF-Connecting-IP": _IP}) == _IP
    assert visitor_ip({"cf-connecting-ip": f"  {_IP}  "}) == _IP
    assert visitor_ip({}) is None
    assert visitor_ip({"CF-Connecting-IP": "  "}) is None


def test_visitor_hash_hides_the_address():
    hashed = visitor_hash(_IP, _PEPPER)
    assert hashed == visitor_hash(_IP, _PEPPER)
    assert hashed != visitor_hash(_IP, "other-pepper")
    assert _IP not in hashed
    assert len(hashed) == 64


def test_public_visitor_hash_needs_pepper_and_address(monkeypatch, capsys):
    monkeypatch.delenv(PEPPER_ENV, raising=False)
    monkeypatch.delenv(VISITOR_IP_ENV, raising=False)
    assert public_visitor_hash({"CF-Connecting-IP": _IP}) is None
    assert "TRIPPY_DEMO_QUOTA_PEPPER is unset" in capsys.readouterr().out

    monkeypatch.setenv(PEPPER_ENV, _PEPPER)
    assert public_visitor_hash({}) is None
    assert "CF-Connecting-IP missing" in capsys.readouterr().out
    assert public_visitor_hash({"CF-Connecting-IP": _IP}) == visitor_hash(
        _IP, _PEPPER
    )

    monkeypatch.setenv(VISITOR_IP_ENV, "127.0.0.1")
    assert public_visitor_hash({}) == visitor_hash("127.0.0.1", _PEPPER)
    assert public_visitor_hash({"CF-Connecting-IP": _IP}) == visitor_hash(
        _IP, _PEPPER
    )


def test_remaining_caption_for_five_one_and_zero():
    assert remaining_caption(5) == "5 questions left | נשארו 5 שאלות"
    assert remaining_caption(1) == "1 question left | נשארה שאלה אחת"
    assert remaining_caption(0) == "No questions left | לא נשארו שאלות"


@pytest.fixture
def quota_conn():
    with psycopg.connect(db_url(), options=SEARCH_PATH) as conn:
        conn.execute("CREATE SCHEMA IF NOT EXISTS experiments")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments.demo_query_quota (
                visitor_hash text PRIMARY KEY,
                query_count integer NOT NULL,
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute("TRUNCATE experiments.demo_query_quota")
        conn.commit()
        yield conn
        conn.rollback()


def test_first_claim_leaves_four(quota_conn):
    visitor = visitor_hash(_IP, _PEPPER)
    decision = claim_query(visitor, cap=_CAP, conn=quota_conn)
    assert decision.allowed
    assert decision.used == 1
    assert decision.remaining == 4
    assert quota_remaining(visitor, cap=_CAP, conn=quota_conn) == 4


def test_sixth_claim_is_denied_without_increment(quota_conn):
    visitor = visitor_hash("198.51.100.4", _PEPPER)
    for _ in range(_CAP):
        assert claim_query(visitor, cap=_CAP, conn=quota_conn).allowed
    denied = claim_query(visitor, cap=_CAP, conn=quota_conn)
    assert not denied.allowed
    assert denied.used == _CAP
    assert denied.remaining == 0
    assert quota_remaining(visitor, cap=_CAP, conn=quota_conn) == 0
    again = claim_query(visitor, cap=_CAP, conn=quota_conn)
    assert not again.allowed
    assert again.used == _CAP
