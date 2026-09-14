"""scrape-reviews change-report markdown for the Actions Summary."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx

from source.scraper.populate_reviews_and_claims import (
    UPSERT_REVIEW_SQL,
    populate_google_reviews,
    store_fetched_reviews,
)
from source.scraper.reviews_report import (
    FetchError,
    NewReview,
    ReviewsRun,
    SkippedSite,
    render_run_report,
    review_preview,
    write_run_report,
)


def test_upsert_sql_marks_inserts_via_xmax():
    assert "RETURNING id, skip_reason, (xmax = 0) AS inserted" in UPSERT_REVIEW_SQL


def test_review_preview_truncates_and_collapses_whitespace():
    assert review_preview("") == "(empty)"
    assert review_preview("hot water") == "hot water"
    long = "x" * 90
    assert review_preview(long) == ("x" * 77) + "..."
    assert review_preview("hot\n  water") == "hot water"


def test_render_run_report_lists_new_reviews_skips_and_errors(tmp_path: Path):
    run = ReviewsRun(
        started_at=datetime(2026, 9, 14, 6, 0, tzinfo=timezone.utc),
        seconds=8.2,
        sites=3,
        reviews_fetched=7,
        reviews_inserted=1,
        reviews_seen=6,
        new_reviews=[
            NewReview(
                site_id=2,
                site_name="Achziv",
                author="Dana",
                rating=4,
                published_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
                preview="hot showers",
            )
        ],
        skipped=[
            SkippedSite(site_id=5, site_name="Masada", reason="no_reviews"),
        ],
        http_errors=[
            FetchError(site_id=7, site_name="Yehiam", message="OVER_QUERY_LIMIT"),
        ],
    )
    text = render_run_report(run)
    assert text.startswith("# scrape-reviews")
    assert "Sites: 3 · fetched: 7 · new: 1 · already stored: 6" in text
    assert "Dana" in text
    assert "hot showers" in text
    assert "Masada" in text
    assert "no_reviews" in text
    assert "OVER_QUERY_LIMIT" in text
    assert "Google Place Details only" in text
    dest = tmp_path / "reviews.md"
    written = write_run_report(text, path=dest)
    assert written == dest
    assert dest.read_text(encoding="utf-8") == text


def test_store_fetched_reviews_records_inserts_on_run():
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.side_effect = [(11, None, True), (12, None, False)]
    conn.cursor.return_value.__enter__.return_value = cur
    run = ReviewsRun(started_at=datetime(2026, 9, 14, tzinfo=timezone.utc))
    result = store_fetched_reviews(
        3,
        {
            "name": "חורשת טל",
            "reviews": [
                {"author": "A", "text": "hot water", "rating": 5},
                {"author": "B", "text": "again", "rating": 4},
            ],
        },
        conn=conn,
        place="חורשת טל",
        run=run,
    )
    assert result == {"campsite_id": 3, "reviews": 2, "claims": 0}
    assert run.reviews_inserted == 1
    assert run.reviews_seen == 1
    assert run.new_reviews[0].author == "A"
    assert run.new_reviews[0].preview == "hot water"


def test_populate_google_reviews_continues_after_http_error():
    conn = MagicMock()
    select_cur = MagicMock()
    select_cur.fetchall.return_value = [
        (1, "חורשת טל", "ChIJ-aaa"),
        (2, "אכזיב", "ChIJ-bbb"),
    ]
    conn.cursor.return_value.__enter__.return_value = select_cur
    client = MagicMock()
    client.get.side_effect = [
        SimpleNamespace(
            json=lambda: {
                "status": "OK",
                "result": {"name": "x", "place_id": "ChIJ-aaa", "reviews": []},
            },
            raise_for_status=lambda: None,
        ),
        SimpleNamespace(
            json=lambda: {
                "status": "OK",
                "result": {"name": "x", "place_id": "ChIJ-aaa", "reviews": []},
            },
            raise_for_status=lambda: None,
        ),
        httpx.HTTPError("503"),
    ]
    populate = MagicMock(return_value={"campsite_id": 1, "reviews": 0, "claims": 0})
    run = ReviewsRun(started_at=datetime(2026, 9, 14, tzinfo=timezone.utc))
    result = populate_google_reviews(
        conn=conn,
        client=client,
        api_key="fake-key",
        pause_seconds=0,
        populate_fn=populate,
        run=run,
    )
    assert [row["campsite_id"] for row in result["sites"]] == [1, 2]
    assert result["sites"][1]["skipped"] == "http_error"
    assert run.http_errors[0].site_id == 2
    assert "503" in run.http_errors[0].message
    conn.commit.assert_called_once()
