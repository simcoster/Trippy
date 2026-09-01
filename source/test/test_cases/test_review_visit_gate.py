"""30B personal-visit gate before claim split. Claims filter is unchanged."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from source.scraper.amenity_enrichment.llm import QWEN_INSTRUCT_30B_MODEL
from source.scraper.populate_reviews_and_claims import (
    SKIP_REASON_NOT_PERSONAL,
    judge_personal_visit,
    populate_reviews_and_claims,
    visit_gate_from_payload,
)


def test_visit_gate_keeps_personal_and_drops_note():
    personal, note = visit_gate_from_payload(
        {"personal_visit": True, "reason": "first-person stay"}
    )
    assert personal is True
    assert note is None


def test_visit_gate_drops_ad():
    personal, note = visit_gate_from_payload(
        {"personal_visit": False, "reason": "brochure dump"}
    )
    assert personal is False
    assert note == "brochure dump"


def test_visit_gate_unreadable_fails_open():
    assert visit_gate_from_payload({}) == (True, None)
    assert visit_gate_from_payload({"personal_visit": "maybe"}) == (True, None)


def test_judge_personal_visit_false_from_model():
    message = SimpleNamespace(
        content=json.dumps(
            {"personal_visit": False, "reason": "history lecture, not a visit"}
        )
    )
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
    )
    personal, note = judge_personal_visit(
        client, {"text": "ווייז : גן לאומי", "rating": 5}, place="יחיעם"
    )
    assert personal is False
    assert "history" in (note or "")
    assert client.chat.completions.create.call_args.kwargs["model"] == (
        QWEN_INSTRUCT_30B_MODEL
    )


def test_judge_personal_visit_parse_error_fails_open():
    message = SimpleNamespace(content="not json")
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
    )
    personal, note = judge_personal_visit(client, {"text": "hot showers"}, place="x")
    assert personal is True
    assert note is None


@patch("source.scraper.populate_reviews_and_claims.register_vector")
def test_populate_does_not_split_skipped_review(_register_vector):
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = (42,)
    conn.cursor.return_value.__enter__.return_value = cur

    judge_msg = SimpleNamespace(
        content=json.dumps({"personal_visit": False, "reason": "ad"})
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
            "name": "גן לאומי מבצר יחיעם",
            "reviews": [{"text": "ווייז : גן לאומי מבצר יחיעם\nלא להחמיץ", "rating": 5}],
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
    embedder.embed.assert_not_called()
    skip_params = [
        call.args[1]
        for call in cur.execute.call_args_list
        if call.args and "skip_reason" in str(call.args[0])
    ]
    assert any(
        params.get("skip_reason") == SKIP_REASON_NOT_PERSONAL for params in skip_params
    )


@patch("source.scraper.populate_reviews_and_claims.register_vector")
def test_populate_splits_after_personal_gate(_register_vector):
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = (7,)
    conn.cursor.return_value.__enter__.return_value = cur

    judge_msg = SimpleNamespace(
        content=json.dumps({"personal_visit": True, "reason": "camped here"})
    )
    split_msg = SimpleNamespace(
        content=json.dumps(
            {
                "claims": [
                    {
                        "text_en": "The showers are clean.",
                        "polarity": "positive",
                        "evidence_span": "מקלחות נקיות",
                        "confidence": 0.9,
                    }
                ]
            }
        )
    )
    chat = MagicMock()
    chat.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=judge_msg)], usage=None),
        SimpleNamespace(choices=[SimpleNamespace(message=split_msg)], usage=None),
    ]
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]

    result = populate_reviews_and_claims(
        1,
        {"name": "חורשת טל", "reviews": [{"text": "המקלחות נקיות", "rating": 4}]},
        conn=conn,
        chat_client=chat,
        embedder=embedder,
    )
    assert result["claims"] == 1
    assert chat.chat.completions.create.call_count == 2
    embedder.embed.assert_called_once()
    skip_clears = [
        call.args[1]
        for call in cur.execute.call_args_list
        if call.args and "skip_reason" in str(call.args[0])
    ]
    assert any(params.get("skip_reason") is None for params in skip_clears)
