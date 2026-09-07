"""Google Place Details review fetch: newest then most_relevant, concatenated."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from source.scraper.populate_reviews_and_claims import (
    DETAILS_FIELDS,
    DETAILS_URL,
    REVIEWS_SORT_MOST_RELEVANT,
    REVIEWS_SORT_NEWEST,
    concat_review_payloads,
    drop_duplicate_reviews,
    fetch_place_details,
    google_review_from_place_review,
    refresh_google_reviews_for_campsite,
    reviews_payload_from_details,
    reviews_sorts_to_fetch,
)

PLACE_ID = "ChIJDUZZZ2-8HhURv7LbSjS_yG0"


def _details(reviews: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        json=lambda: {
            "status": "OK",
            "result": {
                "name": "חורשת טל",
                "place_id": PLACE_ID,
                "reviews": reviews,
            },
        },
        raise_for_status=lambda: None,
    )


def test_reviews_sorts_are_always_newest_then_most_relevant():
    assert reviews_sorts_to_fetch() == [
        REVIEWS_SORT_NEWEST,
        REVIEWS_SORT_MOST_RELEVANT,
    ]
    assert reviews_sorts_to_fetch(most_relevant=False) == [
        REVIEWS_SORT_NEWEST,
        REVIEWS_SORT_MOST_RELEVANT,
    ]


def test_google_review_maps_unix_time():
    published = datetime(2026, 8, 6, 18, 25, 58, tzinfo=timezone.utc)
    mapped = google_review_from_place_review(
        {
            "author_name": "שלומי",
            "rating": 4,
            "time": int(published.timestamp()),
            "text": "מים חמים אין",
        }
    )
    assert mapped["author"] == "שלומי"
    assert mapped["rating"] == 4
    assert mapped["published_utc"] == "2026-08-06T18:25:58+00:00"
    assert mapped["text"] == "מים חמים אין"


def test_reviews_payload_from_details_empty_result():
    assert reviews_payload_from_details({"status": "ZERO_RESULTS"}, reviews_sort="newest") is None
    payload = reviews_payload_from_details(
        {
            "status": "OK",
            "result": {
                "name": "חורשת טל",
                "place_id": PLACE_ID,
                "reviews": [
                    {"author_name": "A", "rating": 5, "time": 1, "text": "ok"},
                    {"author_name": "B", "rating": 1, "time": 2, "text": ""},
                ],
            },
        },
        reviews_sort="newest",
    )
    assert payload is not None
    assert payload["place_id"] == PLACE_ID
    assert payload["reviews_sort"] == "newest"
    assert len(payload["reviews"]) == 2
    assert payload["reviews"][1]["text"] == ""


def test_fetch_place_details_sends_reviews_sort():
    client = MagicMock()
    client.get.return_value = SimpleNamespace(
        json=lambda: {"status": "OK", "result": {"place_id": PLACE_ID}},
        raise_for_status=lambda: None,
    )
    fetch_place_details(client, PLACE_ID, "fake-key", reviews_sort="newest")
    client.get.assert_called_once_with(
        DETAILS_URL,
        params={
            "place_id": PLACE_ID,
            "fields": DETAILS_FIELDS,
            "language": "he",
            "reviews_sort": "newest",
            "key": "fake-key",
        },
    )


def test_refresh_skips_without_place_id():
    populate = MagicMock()
    result = refresh_google_reviews_for_campsite(
        MagicMock(),
        {"id": 1, "name": "חורשת טל", "google_place_id": None},
        client=MagicMock(),
        api_key="fake-key",
        populate_fn=populate,
    )
    assert result["skipped"] == "no_place_id"
    populate.assert_not_called()


def test_drop_duplicate_reviews_when_most_relevant_is_also_newest():
    newest = {"author": "A", "published_utc": "2026-09-01T00:00:00+00:00", "text": "hot water"}
    relevant = {
        "author": "A",
        "published_utc": "2026-09-01T00:00:00+00:00",
        "text": "hot water",
        "rating": 5,
    }
    assert drop_duplicate_reviews([newest, relevant]) == [newest]


def test_concat_review_payloads_newest_then_relevant_dedupes():
    newest = {
        "name": "חורשת טל",
        "place_id": PLACE_ID,
        "reviews": [
            {"author": "A", "published_utc": "1", "text": "new"},
            {"author": "B", "published_utc": "2", "text": "shared"},
        ],
    }
    relevant = {
        "name": "חורשת טל",
        "place_id": PLACE_ID,
        "reviews": [
            {"author": "B", "published_utc": "2", "text": "shared"},
            {"author": "C", "published_utc": "3", "text": "best"},
        ],
    }
    combined = concat_review_payloads([newest, relevant])
    assert combined is not None
    assert [r["text"] for r in combined["reviews"]] == ["new", "shared", "best"]
    assert combined["reviews_sort"] == "newest+most_relevant"


def test_refresh_fetches_both_sorts_concats_and_ingests_once():
    client = MagicMock()
    client.get.side_effect = [
        _details([{"author_name": "A", "rating": 5, "time": 1, "text": "new"}]),
        _details(
            [
                {"author_name": "A", "rating": 5, "time": 1, "text": "new"},
                {"author_name": "C", "rating": 4, "time": 3, "text": "best"},
            ]
        ),
    ]
    populate = MagicMock(return_value={"campsite_id": 1, "reviews": 2, "claims": 0})
    refresh_google_reviews_for_campsite(
        MagicMock(),
        {"id": 1, "name": "חורשת טל", "google_place_id": PLACE_ID},
        client=client,
        api_key="fake-key",
        populate_fn=populate,
    )
    sorts = [call.kwargs["params"]["reviews_sort"] for call in client.get.call_args_list]
    assert sorts == ["newest", "most_relevant"]
    populate.assert_called_once()
    payload = populate.call_args.args[1]
    assert [r["text"] for r in payload["reviews"]] == ["new", "best"]
    assert payload["reviews_sort"] == "newest+most_relevant"


def test_refresh_most_relevant_flag_is_a_noop():
    client = MagicMock()
    client.get.side_effect = [
        _details([]),
        _details([]),
    ]
    populate = MagicMock(return_value={"campsite_id": 1, "reviews": 0, "claims": 0})
    refresh_google_reviews_for_campsite(
        MagicMock(),
        {"id": 1, "name": "חורשת טל", "google_place_id": PLACE_ID},
        most_relevant=True,
        client=client,
        api_key="fake-key",
        populate_fn=populate,
    )
    sorts = [call.kwargs["params"]["reviews_sort"] for call in client.get.call_args_list]
    assert sorts == ["newest", "most_relevant"]
    populate.assert_called_once()
