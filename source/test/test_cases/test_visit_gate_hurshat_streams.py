"""Hurshat Tal streams rant is a guest visit — visit gate must keep it."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_30B_MODEL,
    QWEN_INSTRUCT_MODEL,
)
from source.scraper.populate_reviews_and_claims import (
    VISIT_GATE_SYSTEM,
    populate_reviews_and_claims,
)

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "reviews" / "visit_gate.json"
)
STREAMS = next(
    item
    for item in json.loads(FIXTURE.read_text(encoding="utf-8"))["reviews"]
    if item["id"] == "hurshat_streams"
)

STREAMS_TEXT = (
    "ממש לא תקין ששילמנו כסף מלא ו80% אחוזים מהתעלות במקום  והנחלים "
    'מוובשים וסגורים . ד"א כבר מעל שנה נאמר שזה בשיפוץ תפסיקו לחסוך '
    "ותפתחו את כל הנחלים בשמורה למה רובם יבשים וכל הילדים והאנשים "
    "מצטופפים רק ב 2 נקודות ממש בושה  וחוצפה  !!!"
)

KEPT_CLAIM = {
    "text_en": "80 percent of the channels and streams at the site are dry and closed.",
    "polarity": "negative",
    "evidence_span": "80% אחוזים מהתעלות במקום  והנחלים מוובשים וסגורים",
    "confidence": 0.9,
}


def test_hurshat_streams_is_gold_keep():
    assert STREAMS["relevant"] is True
    assert STREAMS["name"] == "חורשת טל"
    assert STREAMS["text"] == STREAMS_TEXT


def test_visit_gate_prompt_keeps_paid_visit_stream_complaints():
    assert "80% of the channels and streams are dry" in VISIT_GATE_SYSTEM
    assert "they were at the site" in VISIT_GATE_SYSTEM


@patch("source.scraper.populate_reviews_and_claims.register_vector")
def test_hurshat_streams_review_is_split_and_embedded(_register_vector):
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = (8,)
    conn.cursor.return_value.__enter__.return_value = cur

    chat = MagicMock()
    chat.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "personal_visit": True,
                                "reason": "paid visit; witnessed dry streams and crowding",
                            }
                        )
                    )
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({"claims": [KEPT_CLAIM]}))
                )
            ],
            usage=None,
        ),
    ]
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]

    result = populate_reviews_and_claims(
        1,
        {
            "name": STREAMS["name"],
            "reviews": [{"text": STREAMS["text"], "rating": STREAMS.get("rating")}],
        },
        conn=conn,
        chat_client=chat,
        embedder=embedder,
    )
    assert result["claims"] == 1
    models = [
        call.kwargs["model"] for call in chat.chat.completions.create.call_args_list
    ]
    assert models == [QWEN_INSTRUCT_30B_MODEL, QWEN_INSTRUCT_MODEL]
    user = chat.chat.completions.create.call_args_list[0].kwargs["messages"][1][
        "content"
    ]
    sent = json.loads(user)
    assert sent["review"]["text"] == STREAMS["text"]
    embedder.embed.assert_called_once()
    assert embedder.embed.call_args.args[0] == [KEPT_CLAIM["text_en"]]
    skip_clears = [
        call.args[1]
        for call in cur.execute.call_args_list
        if call.args and "skip_reason" in str(call.args[0])
    ]
    assert any(params.get("skip_reason") is None for params in skip_clears)
    assert any(
        "INSERT INTO claims" in str(call.args[0])
        for call in cur.execute.call_args_list
        if call.args
    )
