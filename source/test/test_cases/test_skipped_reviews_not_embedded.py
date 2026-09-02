"""Skipped reviews stay off the 235B split and embed path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_30B_MODEL,
    QWEN_INSTRUCT_MODEL,
)
from source.scraper.populate_reviews_and_claims import (
    DELETE_CLAIMS_FOR_REVIEW_SQL,
    INSERT_CLAIM_SQL,
    SKIP_REASON_NOT_PERSONAL,
    populate_reviews_and_claims,
    upsert_review,
)

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "reviews" / "visit_gate.json"
)
_CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["reviews"]
AD = next(item for item in _CASES if item["id"] == "yehiam_ad")
GUEST = next(item for item in _CASES if item["id"] == "hurshat_showers")

KEPT_CLAIM = {
    "text_en": "The showers are clean.",
    "polarity": "positive",
    "evidence_span": "מקלחות נקיות",
    "confidence": 0.9,
}


def _sql(call) -> str:
    return str(call.args[0]) if call.args else ""


def _conn_cur(fetchone):
    conn = MagicMock()
    cur = MagicMock()
    if isinstance(fetchone, list):
        cur.fetchone.side_effect = fetchone
    else:
        cur.fetchone.return_value = fetchone
    conn.cursor.return_value.__enter__.return_value = cur
    return conn, cur


def _chat_json(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
        ],
        usage=None,
    )


def _chat_models(chat) -> list[str]:
    return [
        call.kwargs["model"] for call in chat.chat.completions.create.call_args_list
    ]


def _claim_insert_review_ids(cur) -> list[int]:
    return [
        call.args[1]["review_id"]
        for call in cur.execute.call_args_list
        if call.args and "INSERT INTO claims" in _sql(call)
    ]


@patch("source.scraper.populate_reviews_and_claims.register_vector")
def test_already_skipped_review_is_not_split_or_embedded(_register_vector):
    conn, cur = _conn_cur((42, SKIP_REASON_NOT_PERSONAL))
    chat = MagicMock()
    embedder = MagicMock()

    result = populate_reviews_and_claims(
        1,
        {"name": AD["name"], "reviews": [{"text": AD["text"], "rating": AD["rating"]}]},
        conn=conn,
        chat_client=chat,
        embedder=embedder,
    )
    assert result["claims"] == 0
    chat.chat.completions.create.assert_not_called()
    embedder.embed.assert_not_called()
    executed = [_sql(call) for call in cur.execute.call_args_list]
    assert any("DELETE FROM claims" in sql for sql in executed)
    assert not any("INSERT INTO claims" in sql for sql in executed)


@patch("source.scraper.populate_reviews_and_claims.register_vector")
def test_gate_skip_does_not_insert_or_embed_claims(_register_vector):
    conn, cur = _conn_cur((42,))
    chat = MagicMock()
    chat.chat.completions.create.return_value = _chat_json(
        {"personal_visit": False, "reason": "ad"}
    )
    embedder = MagicMock()

    result = populate_reviews_and_claims(
        1,
        {"name": AD["name"], "reviews": [{"text": AD["text"], "rating": AD["rating"]}]},
        conn=conn,
        chat_client=chat,
        embedder=embedder,
    )
    assert result["claims"] == 0
    assert chat.chat.completions.create.call_count == 1
    assert _chat_models(chat) == [QWEN_INSTRUCT_30B_MODEL]
    embedder.embed.assert_not_called()
    executed = [_sql(call) for call in cur.execute.call_args_list]
    assert any(sql.strip() == DELETE_CLAIMS_FOR_REVIEW_SQL.strip() for sql in executed)
    assert not any(sql.strip() == INSERT_CLAIM_SQL.strip() for sql in executed)


@patch("source.scraper.populate_reviews_and_claims.register_vector")
def test_skipped_review_in_batch_is_not_split_or_embedded(_register_vector):
    conn, cur = _conn_cur([(10,), (20,)])
    chat = MagicMock()
    chat.chat.completions.create.side_effect = [
        _chat_json({"personal_visit": False, "reason": "ad"}),
        _chat_json({"personal_visit": True, "reason": "camped here"}),
        _chat_json({"claims": [KEPT_CLAIM]}),
    ]
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]

    result = populate_reviews_and_claims(
        1,
        {
            "name": AD["name"],
            "reviews": [
                {"text": AD["text"], "rating": AD["rating"]},
                {"text": GUEST["text"], "rating": GUEST["rating"]},
            ],
        },
        conn=conn,
        chat_client=chat,
        embedder=embedder,
    )
    assert result["claims"] == 1
    assert _chat_models(chat) == [
        QWEN_INSTRUCT_30B_MODEL,
        QWEN_INSTRUCT_30B_MODEL,
        QWEN_INSTRUCT_MODEL,
    ]
    embedder.embed.assert_called_once()
    assert embedder.embed.call_args.args[0] == [KEPT_CLAIM["text_en"]]
    assert _claim_insert_review_ids(cur) == [20]


@patch("source.scraper.populate_reviews_and_claims.register_vector")
def test_already_skipped_in_batch_is_not_split_or_embedded(_register_vector):
    conn, cur = _conn_cur([(10, SKIP_REASON_NOT_PERSONAL), (20,)])
    chat = MagicMock()
    chat.chat.completions.create.side_effect = [
        _chat_json({"personal_visit": True, "reason": "camped here"}),
        _chat_json({"claims": [KEPT_CLAIM]}),
    ]
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]

    result = populate_reviews_and_claims(
        1,
        {
            "name": AD["name"],
            "reviews": [
                {"text": AD["text"], "rating": AD["rating"]},
                {"text": GUEST["text"], "rating": GUEST["rating"]},
            ],
        },
        conn=conn,
        chat_client=chat,
        embedder=embedder,
    )
    assert result["claims"] == 1
    assert chat.chat.completions.create.call_count == 2
    assert _chat_models(chat) == [QWEN_INSTRUCT_30B_MODEL, QWEN_INSTRUCT_MODEL]
    embedder.embed.assert_called_once()
    assert embedder.embed.call_args.args[0] == [KEPT_CLAIM["text_en"]]
    assert _claim_insert_review_ids(cur) == [20]


def test_upsert_review_returns_existing_skip_reason():
    cur = MagicMock()
    cur.fetchone.return_value = (9, SKIP_REASON_NOT_PERSONAL)
    review_id, skip = upsert_review(
        cur,
        campsite_id=1,
        review={"source": "google", "text": "ad", "author": None, "published_at": None},
    )
    assert review_id == 9
    assert skip == SKIP_REASON_NOT_PERSONAL
    sql = _sql(cur.execute.call_args)
    assert "RETURNING id, skip_reason" in sql


def test_upsert_review_returns_none_skip_for_new_row():
    cur = MagicMock()
    cur.fetchone.return_value = (9, None)
    review_id, skip = upsert_review(
        cur,
        campsite_id=1,
        review={"source": "google", "text": "ok", "author": None, "published_at": None},
    )
    assert review_id == 9
    assert skip is None


@patch("source.agent.search.psycopg.connect")
@patch("source.agent.search.register_vector")
def test_search_review_claims_excludes_skipped_reviews(_register_vector, connect):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    cur = MagicMock()
    cur.fetchall.return_value = []
    conn.cursor.return_value.__enter__.return_value = cur
    connect.return_value = conn

    from source.agent.search import search_review_claims

    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://mock"}):
        search_review_claims("hot showers", embedding="[0.1]")

    sql = _sql(cur.execute.call_args)
    assert "r.skip_reason IS NULL" in sql
