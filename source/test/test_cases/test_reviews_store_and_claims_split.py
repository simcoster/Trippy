"""scrape-reviews stores rows; populate-claims classifies; clear-claims resets flag."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.clear_claims import CLEAR_CLAIMS_SQL, RESET_RELEVANT_SQL, clear_claims
from source.scraper.populate_claims import (
    MARK_EMPTY_IRRELEVANT_SQL,
    SELECT_UNCLASSIFIED_SQL,
    fetch_unclassified_reviews,
    mark_empty_reviews_irrelevant,
    populate_claims_for_rows,
)
from source.scraper.populate_reviews_and_claims import (
    UPSERT_REVIEW_SQL,
    refresh_google_reviews_for_campsite,
    store_fetched_reviews,
)


def test_refresh_defaults_to_store_not_claim_split():
    client = MagicMock()
    client.get.side_effect = [
        MagicMock(
            json=lambda: {
                "status": "OK",
                "result": {
                    "name": "חורשת טל",
                    "place_id": "ChIJ",
                    "reviews": [
                        {"author_name": "A", "rating": 5, "time": 1, "text": "hot"}
                    ],
                },
            },
            raise_for_status=lambda: None,
        ),
        MagicMock(
            json=lambda: {
                "status": "OK",
                "result": {
                    "name": "חורשת טל",
                    "place_id": "ChIJ",
                    "reviews": [],
                },
            },
            raise_for_status=lambda: None,
        ),
    ]
    with patch(
        "source.scraper.populate_reviews_and_claims.store_fetched_reviews"
    ) as store:
        store.return_value = {"campsite_id": 1, "reviews": 1, "claims": 0}
        refresh_google_reviews_for_campsite(
            MagicMock(),
            {"id": 1, "name": "חורשת טל", "google_place_id": "ChIJ"},
            client=client,
            api_key="fake-key",
        )
        store.assert_called_once()


def test_store_fetched_reviews_upserts_without_llm():
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = (1, None)
    conn.cursor.return_value.__enter__.return_value = cur
    chat = MagicMock()

    result = store_fetched_reviews(
        3,
        {
            "name": "חורשת טל",
            "reviews": [
                {"author": "A", "text": "hot water", "rating": 5},
            ],
        },
        conn=conn,
        chat_client=chat,
    )
    assert result == {"campsite_id": 3, "reviews": 1, "claims": 0}
    chat.chat.completions.create.assert_not_called()
    sql = str(cur.execute.call_args.args[0])
    assert "INSERT INTO reviews" in sql
    assert "is_relevant = CASE" in UPSERT_REVIEW_SQL
    assert "WHEN btrim(EXCLUDED.text) = '' THEN FALSE" in UPSERT_REVIEW_SQL
    assert "WHEN btrim(%(text)s) = '' THEN FALSE" in UPSERT_REVIEW_SQL
    assert not any(
        "INSERT INTO claims" in str(call.args[0])
        for call in cur.execute.call_args_list
        if call.args
    )


def test_fetch_unclassified_reviews_filters_is_relevant_null():
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchall.return_value = [
        (9, 3, "google", "A", 5, "hot water", None, None, "חורשת טל"),
    ]
    conn.cursor.return_value.__enter__.return_value = cur
    rows = fetch_unclassified_reviews(conn, campsite_id=3)
    sql = cur.execute.call_args.args[0]
    params = cur.execute.call_args.args[1]
    assert sql.strip() == SELECT_UNCLASSIFIED_SQL.strip()
    assert "is_relevant IS NULL" in sql
    assert params["campsite_id"] == 3
    assert rows[0]["id"] == 9


@patch("source.scraper.populate_claims.register_vector")
@patch("source.scraper.populate_claims.split_one_review")
@patch("source.scraper.populate_claims.judge_personal_visit")
def test_populate_claims_sets_is_relevant_and_skips_ads(
    judge, split, _register
):
    judge.side_effect = [(False, "ad"), (True, None)]
    split.return_value = [
        {
            "text_en": "The showers are clean.",
            "polarity": "positive",
            "evidence_span": "מקלחות",
            "confidence": 0.9,
        }
    ]
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]
    rows = [
        {
            "id": 10,
            "campsite_id": 1,
            "source": "google",
            "author": "x",
            "rating": 5,
            "text": "ad copy",
            "published_at": None,
            "place": "יחיעם",
        },
        {
            "id": 11,
            "campsite_id": 1,
            "source": "google",
            "author": "y",
            "rating": 4,
            "text": "camped here",
            "published_at": None,
            "place": "יחיעם",
        },
    ]
    result = populate_claims_for_rows(
        conn, rows, chat_client=MagicMock(), embedder=embedder
    )
    assert result["reviews"] == 2
    assert result["skipped"] == 1
    assert result["claims"] == 1
    relevance = [
        call.args[1]["is_relevant"]
        for call in cur.execute.call_args_list
        if call.args and "is_relevant" in str(call.args[0]) and isinstance(call.args[1], dict)
        and "is_relevant" in call.args[1]
        and call.args[1].get("review_id") in (10, 11)
    ]
    assert False in relevance
    assert True in relevance
    conn.commit.assert_called_once()


@patch("source.scraper.populate_claims.register_vector")
@patch("source.scraper.populate_claims.split_one_review")
@patch("source.scraper.populate_claims.judge_personal_visit")
def test_populate_claims_skips_already_classified_reviews(
    judge, split, _register
):
    judge.return_value = (True, None)
    split.return_value = [
        {
            "text_en": "The showers are clean.",
            "polarity": "positive",
            "evidence_span": "מקלחות",
            "confidence": 0.9,
        }
    ]
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]
    rows = [
        {
            "id": 10,
            "campsite_id": 1,
            "source": "google",
            "author": "x",
            "rating": 5,
            "text": "already gated",
            "published_at": None,
            "is_relevant": False,
            "place": "יחיעם",
        },
        {
            "id": 11,
            "campsite_id": 1,
            "source": "google",
            "author": "y",
            "rating": 4,
            "text": "camped here",
            "published_at": None,
            "is_relevant": None,
            "place": "יחיעם",
        },
    ]
    result = populate_claims_for_rows(
        conn, rows, chat_client=MagicMock(), embedder=embedder
    )
    assert result["reviews"] == 1
    judge.assert_called_once()
    split.assert_called_once()


@patch("source.scraper.populate_claims.register_vector")
@patch("source.scraper.populate_claims.split_one_review")
@patch("source.scraper.populate_claims.judge_personal_visit")
def test_populate_claims_commits_after_each_campsite(judge, split, _register):
    judge.return_value = (True, None)
    split.return_value = [
        {
            "text_en": "The showers are clean.",
            "polarity": "positive",
            "evidence_span": "מקלחות",
            "confidence": 0.9,
        }
    ]
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]
    rows = [
        {
            "id": 10,
            "campsite_id": 1,
            "source": "google",
            "author": "x",
            "rating": 5,
            "text": "camped at one",
            "published_at": None,
            "place": "יחיעם",
        },
        {
            "id": 11,
            "campsite_id": 2,
            "source": "google",
            "author": "y",
            "rating": 4,
            "text": "camped at two",
            "published_at": None,
            "place": "חורשת טל",
        },
    ]
    populate_claims_for_rows(
        conn, rows, chat_client=MagicMock(), embedder=embedder
    )
    assert conn.commit.call_count == 2
    assert conn.rollback.call_count == 0


def test_mark_empty_reviews_sets_is_relevant_false():
    conn = MagicMock()
    cur = MagicMock()
    cur.rowcount = 3
    conn.cursor.return_value.__enter__.return_value = cur
    n = mark_empty_reviews_irrelevant(conn, campsite_id=2)
    sql = cur.execute.call_args.args[0]
    params = cur.execute.call_args.args[1]
    assert sql.strip() == MARK_EMPTY_IRRELEVANT_SQL.strip()
    assert "is_relevant = FALSE" in sql
    assert "btrim(text) = ''" in sql
    assert params["campsite_id"] == 2
    assert n == 3


@patch("source.scraper.populate_claims.register_vector")
@patch("source.scraper.populate_claims.split_one_review")
@patch("source.scraper.populate_claims.judge_personal_visit")
def test_populate_claims_empty_text_is_not_relevant(judge, split, _register):
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    rows = [
        {
            "id": 10,
            "campsite_id": 1,
            "source": "google",
            "author": "x",
            "rating": 5,
            "text": "   ",
            "published_at": None,
            "place": "יחיעם",
        },
    ]
    result = populate_claims_for_rows(
        conn, rows, chat_client=MagicMock(), embedder=MagicMock()
    )
    assert result["reviews"] == 1
    assert result["skipped"] == 1
    assert result["claims"] == 0
    judge.assert_not_called()
    split.assert_not_called()
    relevance = [
        call.args[1]["is_relevant"]
        for call in cur.execute.call_args_list
        if call.args
        and isinstance(call.args[1], dict)
        and call.args[1].get("review_id") == 10
        and "is_relevant" in call.args[1]
    ]
    assert relevance == [False]


def test_clear_claims_deletes_claims_and_nulls_is_relevant_only():
    conn = MagicMock()
    cur = MagicMock()
    cur.rowcount = 4
    conn.cursor.return_value.__enter__.return_value = cur
    clear_claims(conn)
    executed = [str(call.args[0]) for call in cur.execute.call_args_list]
    assert CLEAR_CLAIMS_SQL in executed
    assert RESET_RELEVANT_SQL in executed
    assert not any("DELETE FROM reviews" in sql for sql in executed)
    assert not any("TRUNCATE" in sql for sql in executed)
