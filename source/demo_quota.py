"""Lifetime query cap for the public Streamlit demo.

The visitor is Cloudflare's CF-Connecting-IP. Postgres stores a hash of
pepper and that address, never the address itself. The count does not reset.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass

import psycopg

from db.connect import connect

DEFAULT_CAP = 5
CAP_ENV = "TRIPPY_DEMO_QUERY_CAP"
PEPPER_ENV = "TRIPPY_DEMO_QUOTA_PEPPER"
VISITOR_IP_ENV = "TRIPPY_DEMO_VISITOR_IP"
_IP_HEADER = "cf-connecting-ip"

QUOTA_USED = "You've used your questions. | ניצלת את השאלות."

_CLAIM_SQL = """
INSERT INTO demo_query_quota (visitor_hash, query_count)
VALUES (%(hash)s, 1)
ON CONFLICT (visitor_hash) DO UPDATE
SET query_count = demo_query_quota.query_count + 1,
    updated_at = now()
WHERE demo_query_quota.query_count < %(cap)s
RETURNING query_count
"""

_COUNT_SQL = """
SELECT query_count FROM demo_query_quota WHERE visitor_hash = %(hash)s
"""


@dataclass(frozen=True)
class QuotaDecision:
    allowed: bool
    used: int
    cap: int

    @property
    def remaining(self) -> int:
        return max(0, self.cap - self.used)


def query_cap() -> int:
    """`TRIPPY_DEMO_QUERY_CAP` when it is a positive integer, else 5."""
    raw = (os.environ.get(CAP_ENV) or "").strip()
    if not raw:
        return DEFAULT_CAP
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CAP
    if value < 1:
        return DEFAULT_CAP
    return value


def quota_pepper() -> str | None:
    pepper = (os.environ.get(PEPPER_ENV) or "").strip()
    return pepper or None


def visitor_ip(headers: Mapping[str, str]) -> str | None:
    """Cloudflare connecting address, or None when the header is absent."""
    for key, value in headers.items():
        if key.casefold() != _IP_HEADER:
            continue
        text = (value or "").strip()
        return text or None
    return None


def visitor_hash(ip: str, pepper: str) -> str:
    return hashlib.sha256(f"{pepper}:{ip}".encode()).hexdigest()


def local_visitor_ip() -> str | None:
    """Stand-in address for a laptop run that has no Cloudflare header."""
    value = (os.environ.get(VISITOR_IP_ENV) or "").strip()
    return value or None


def public_visitor_hash(headers: Mapping[str, str]) -> str | None:
    """Hashed visitor address, or None when this request cannot be capped.

    Cloudflare's header wins. `TRIPPY_DEMO_VISITOR_IP` is used only when
    that header is absent, so a local public UI can exercise the cap.
    """
    pepper = quota_pepper()
    if not pepper:
        print("error: TRIPPY_DEMO_QUOTA_PEPPER is unset", flush=True)
        return None
    ip = visitor_ip(headers)
    if not ip:
        ip = local_visitor_ip()
        if ip:
            print("demo quota using TRIPPY_DEMO_VISITOR_IP", flush=True)
    if not ip:
        print("error: CF-Connecting-IP missing", flush=True)
        return None
    return visitor_hash(ip, pepper)


def remaining_caption(remaining: int) -> str:
    if remaining <= 0:
        return "No questions left | לא נשארו שאלות"
    if remaining == 1:
        return "1 question left | נשארה שאלה אחת"
    return f"{remaining} questions left | נשארו {remaining} שאלות"


def quota_remaining(
    visitor_hash: str,
    *,
    cap: int | None = None,
    conn: psycopg.Connection | None = None,
) -> int:
    """Questions still allowed. No row means the full cap."""
    limit = query_cap() if cap is None else cap
    if conn is None:
        with connect() as connection:
            return _remaining(connection, visitor_hash, limit)
    return _remaining(conn, visitor_hash, limit)


def claim_query(
    visitor_hash: str,
    *,
    cap: int | None = None,
    conn: psycopg.Connection | None = None,
) -> QuotaDecision:
    """Count one question when the visitor is still under the cap.

    A rejected claim does not increment. Opening our own connection commits;
    a passed connection is left for the caller.
    """
    limit = query_cap() if cap is None else cap
    if conn is None:
        with connect() as connection:
            return _claim(connection, visitor_hash, limit)
    return _claim(conn, visitor_hash, limit)


def _remaining(conn: psycopg.Connection, visitor_hash: str, cap: int) -> int:
    row = conn.execute(_COUNT_SQL, {"hash": visitor_hash}).fetchone()
    used = int(row[0]) if row else 0
    return max(0, cap - used)


def _claim(conn: psycopg.Connection, visitor_hash: str, cap: int) -> QuotaDecision:
    row = conn.execute(_CLAIM_SQL, {"hash": visitor_hash, "cap": cap}).fetchone()
    if row is None:
        current = conn.execute(_COUNT_SQL, {"hash": visitor_hash}).fetchone()
        used = int(current[0]) if current else cap
        return QuotaDecision(allowed=False, used=used, cap=cap)
    used = int(row[0])
    return QuotaDecision(allowed=True, used=used, cap=cap)
