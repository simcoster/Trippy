"""Invalid JSON escapes in a recommend stream must not crash."""

from __future__ import annotations

from types import SimpleNamespace

from source.agent.recommender import (
    parse_recommender_payload,
    recommend_from_payload,
)


def _fit():
    return {
        "campsite_id": 37,
        "campsite": "אכזיב צפון",
        "accommodation_type": "אוהל",
        "start": "2026-09-10",
        "end": "2026-09-11",
        "price_per_night": 150,
    }


class _StreamChat:
    def __init__(self, raw: str, *, width: int = 8) -> None:
        self.raw = raw
        self.width = width

    def stream(self, messages, **kwargs):
        for i in range(0, len(self.raw), self.width):
            yield SimpleNamespace(content=self.raw[i : i + self.width])
        yield SimpleNamespace(
            content="",
            usage_metadata={"input_tokens": 3, "output_tokens": 2},
        )


_BAD = (
    '{"recommendations": [{"campsite_id": 37,'
    '"accommodation_type": "אוהל",'
    '"start": "2026-09-10", "end": "2026-09-11",'
    r'"why": "יש \pitch לאוהלים"}], "empty": null}'
)


def test_parse_payload_keeps_rec_after_invalid_escape():
    parsed = parse_recommender_payload(_BAD)
    assert parsed["recommendations"]
    assert "pitch" in parsed["recommendations"][0]["why"]


def test_stream_invalid_escape_does_not_raise():
    result = recommend_from_payload(
        "ליד הים",
        {"constraints": {}, "fits": [_fit()]},
        chat=_StreamChat(_BAD),
    )
    assert len(result.recommendations) == 1
    assert "אכזיב צפון" in result.text
