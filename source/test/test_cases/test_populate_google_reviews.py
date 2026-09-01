"""CLI populate_google_reviews: Places only, no JSON files."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from source.scraper.populate_reviews_and_claims import (
    fetch_sites_for_reviews,
    populate_google_reviews,
)


def test_fetch_sites_for_reviews_requires_place_id():
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchall.return_value = [
        (1, "חורשת טל", "ChIJDUZZZ2-8HhURv7LbSjS_yG0"),
    ]
    conn.cursor.return_value.__enter__.return_value = cur
    rows = fetch_sites_for_reviews(conn, campsite_id=1, limit=5)
    sql = cur.execute.call_args.args[0]
    params = cur.execute.call_args.args[1]
    assert "google_place_id IS NOT NULL" in sql
    assert params["campsite_id"] == 1
    assert params["limit"] == 5
    assert rows[0]["google_place_id"] == "ChIJDUZZZ2-8HhURv7LbSjS_yG0"


def test_populate_google_reviews_ingests_each_site_from_places():
    conn = MagicMock()
    select_cur = MagicMock()
    select_cur.fetchall.return_value = [
        (1, "חורשת טל", "ChIJ-aaa"),
        (2, "אכזיב", "ChIJ-bbb"),
    ]
    conn.cursor.return_value.__enter__.return_value = select_cur

    client = MagicMock()
    client.get.return_value = SimpleNamespace(
        json=lambda: {
            "status": "OK",
            "result": {
                "name": "x",
                "place_id": "ChIJ-aaa",
                "reviews": [],
            },
        },
        raise_for_status=lambda: None,
    )
    populate = MagicMock(return_value={"campsite_id": 1, "reviews": 0, "claims": 0})
    result = populate_google_reviews(
        conn=conn,
        client=client,
        api_key="fake-key",
        pause_seconds=0,
        populate_fn=populate,
    )
    assert [row["campsite_id"] for row in result["sites"]] == [1, 2]
    assert populate.call_count == 2
    conn.commit.assert_called_once()
