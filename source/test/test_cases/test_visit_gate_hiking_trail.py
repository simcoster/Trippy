"""Hiking-trail reviews are visit-gate skips, not campsite-stay claims."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from source.scraper.amenity_enrichment.llm import QWEN_INSTRUCT_30B_MODEL
from source.scraper.populate_reviews_and_claims import (
    SKIP_REASON_NOT_PERSONAL,
    VISIT_GATE_SYSTEM,
    populate_reviews_and_claims,
)

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "reviews" / "visit_gate.json"
)
TRAIL = next(
    item
    for item in json.loads(FIXTURE.read_text(encoding="utf-8"))["reviews"]
    if item["id"] == "mishmar_ha_carmel_nahl_oren_trail"
)


def test_nahl_oren_trail_is_gold_skip():
    assert TRAIL["relevant"] is False
    assert TRAIL["name"] == "חניון לילה חוות משמר הכרמל"
    assert TRAIL["text"] == (
        "מסלול נחל אורן מחניון האגם עד לעין אלון ובחזרה, כולל עליה לתצפית "
        "מחוות משמר הכרמל. טיול בדרגת קושי קל-בינוני. חצי מהמסלול מוצל "
        "ומתאים לטיול גם בימי הקיץ. בעונת הקיץ המסלול יבש."
    )


def test_visit_gate_prompt_drops_hiking_trail_writeups():
    assert "hiking-trail" in VISIT_GATE_SYSTEM
    assert "מסלול" in VISIT_GATE_SYSTEM
    assert "trail guides" in VISIT_GATE_SYSTEM


@patch("source.scraper.populate_reviews_and_claims.register_vector")
def test_nahl_oren_trail_review_is_not_split_or_embedded(_register_vector):
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = (55,)
    conn.cursor.return_value.__enter__.return_value = cur

    judge_msg = SimpleNamespace(
        content=json.dumps(
            {
                "personal_visit": False,
                "reason": "hiking trail writeup, not a campsite stay",
            }
        )
    )
    chat = MagicMock()
    chat.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=judge_msg)],
        usage=None,
    )
    embedder = MagicMock()

    result = populate_reviews_and_claims(
        1,
        {
            "name": TRAIL["name"],
            "reviews": [{"text": TRAIL["text"], "rating": TRAIL.get("rating")}],
        },
        conn=conn,
        chat_client=chat,
        embedder=embedder,
    )
    assert result["claims"] == 0
    assert chat.chat.completions.create.call_count == 1
    models = [
        call.kwargs["model"] for call in chat.chat.completions.create.call_args_list
    ]
    assert models == [QWEN_INSTRUCT_30B_MODEL]
    system = chat.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert system == VISIT_GATE_SYSTEM
    user = chat.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert TRAIL["text"] in user
    embedder.embed.assert_not_called()
    skip_params = [
        call.args[1]
        for call in cur.execute.call_args_list
        if call.args and "skip_reason" in str(call.args[0])
    ]
    assert any(
        params.get("skip_reason") == SKIP_REASON_NOT_PERSONAL for params in skip_params
    )
    assert not any(
        "INSERT INTO claims" in str(call.args[0])
        for call in cur.execute.call_args_list
        if call.args
    )
