"""Legacy Places Text Search → campsites.google_place_id (first hit)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from source.scraper.populate_google_place_id import (
    TEXTSEARCH_URL,
    fetch_campsites,
    first_place_from_textsearch,
    populate_google_place_ids,
    store_google_place_id,
    textsearch,
)

ACHZIV_CAMP = {
    "place_id": "ChIJ6blDQ_rRHRUR5nXf0SggXV0",
    "name": "חניון לילה אכזיב",
    "rating": 3.9,
    "user_ratings_total": 32,
    "types": ["campground"],
}
ACHZIV_PARK = {
    "place_id": "ChIJparkAchzivXXXXXXXXXXXX",
    "name": "גן לאומי אכזיב",
    "types": ["park"],
}


def test_first_place_takes_only_the_first_hit():
    hit = first_place_from_textsearch(
        {"status": "OK", "results": [ACHZIV_CAMP, ACHZIV_PARK]}
    )
    assert hit is not None
    assert hit["place_id"] == ACHZIV_CAMP["place_id"]
    assert hit["name"] == ACHZIV_CAMP["name"]
    assert hit["n_hits"] == 2


def test_first_place_empty_or_zero_results_is_none():
    assert first_place_from_textsearch({"status": "ZERO_RESULTS", "results": []}) is None
    assert first_place_from_textsearch({"status": "OK", "results": []}) is None
    assert first_place_from_textsearch({"status": "REQUEST_DENIED"}) is None
    assert first_place_from_textsearch({"status": "OK", "results": [{}]}) is None


def test_textsearch_sends_hebrew_israel_query():
    client = MagicMock()
    client.get.return_value = SimpleNamespace(
        json=lambda: {"status": "OK", "results": [ACHZIV_CAMP]},
        raise_for_status=lambda: None,
    )
    body = textsearch(client, "חניון לילה גן לאומי אכזיב", "fake-key")
    assert body["status"] == "OK"
    client.get.assert_called_once_with(
        TEXTSEARCH_URL,
        params={
            "query": "חניון לילה גן לאומי אכזיב",
            "language": "he",
            "region": "il",
            "key": "fake-key",
        },
    )


def test_fetch_campsites_skips_filled_unless_force():
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchall.return_value = []
    conn.cursor.return_value.__enter__.return_value = cur
    fetch_campsites(conn, force=False, limit=5)
    assert cur.execute.call_args.args[1]["force"] is False
    fetch_campsites(conn, force=True, campsite_id=2)
    params = cur.execute.call_args.args[1]
    assert params["force"] is True
    assert params["campsite_id"] == 2


def test_store_google_place_id_updates_row():
    cur = MagicMock()
    cur.fetchone.return_value = (2, "חניון לילה גן לאומי אכזיב", ACHZIV_CAMP["place_id"])
    row = store_google_place_id(
        cur, campsite_id=2, place_id=ACHZIV_CAMP["place_id"]
    )
    assert row == (2, "חניון לילה גן לאומי אכזיב", ACHZIV_CAMP["place_id"])
    sql = cur.execute.call_args.args[0]
    params = cur.execute.call_args.args[1]
    assert "google_place_id" in sql
    assert params == {"id": 2, "google_place_id": ACHZIV_CAMP["place_id"]}


def test_populate_writes_first_hit_and_skips_empty():
    client = MagicMock()
    ok = SimpleNamespace(
        json=lambda: {"status": "OK", "results": [ACHZIV_CAMP, ACHZIV_PARK]},
        raise_for_status=lambda: None,
    )
    empty = SimpleNamespace(
        json=lambda: {"status": "ZERO_RESULTS", "results": []},
        raise_for_status=lambda: None,
    )
    client.get.side_effect = [ok, empty]

    conn = MagicMock()
    select_cur = MagicMock()
    select_cur.fetchall.return_value = [
        (2, "חניון לילה גן לאומי אכזיב", None),
        (99, "אין כזה", None),
    ]
    update_cur = MagicMock()
    update_cur.fetchone.return_value = (
        2,
        "חניון לילה גן לאומי אכזיב",
        ACHZIV_CAMP["place_id"],
    )
    conn.cursor.return_value.__enter__.side_effect = [select_cur, update_cur]

    result = populate_google_place_ids(
        conn=conn,
        client=client,
        api_key="fake-key",
        pause_seconds=0,
    )
    assert [row["google_place_id"] for row in result["updated"]] == [
        ACHZIV_CAMP["place_id"]
    ]
    assert result["updated"][0]["n_hits"] == 2
    assert result["skipped"][0]["id"] == 99
    conn.commit.assert_called_once()
