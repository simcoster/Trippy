"""Availability scrape drops nights whose check-in is before today."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from source.scraper.availability_report import PastNightsDeleted
from source.scraper.populate_availability import delete_past_nights


def test_delete_past_nights_cuts_on_start_date_before_today():
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    counts = iter([4, 2])

    def execute(sql, params=None):
        cur.rowcount = next(counts)

    cur.execute.side_effect = execute

    result = delete_past_nights(conn, before=date(2026, 9, 14))

    assert result == PastNightsDeleted(availability_rows=4, hash_rows=2)
    availability_sql, availability_params = cur.execute.call_args_list[0].args
    hash_sql, hash_params = cur.execute.call_args_list[1].args
    assert "DELETE FROM availability" in availability_sql
    assert "start_date < %(before)s" in availability_sql
    assert availability_params["before"] == date(2026, 9, 14)
    assert "DELETE FROM booking_page_hashes" in hash_sql
    assert hash_params["before"] == date(2026, 9, 14)
