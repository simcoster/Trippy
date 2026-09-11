"""Recommend streams tokens and paints the spoken reply, not raw JSON."""

from __future__ import annotations

from types import SimpleNamespace

from source.agent.recommender import listen_recommend_text, recommend_from_payload
from source.scraper.amenity_enrichment.llm import collect_llm_usage


def _fit():
    return {
        "campsite_id": 37,
        "campsite": "אכזיב צפון",
        "accommodation_type": "אוהל",
        "start": "2026-09-10",
        "end": "2026-09-11",
        "price_per_night": 150,
    }


_JSON = (
    '{"recommendations": [{"campsite_id": 37,'
    '"accommodation_type": "אוהל",'
    '"start": "2026-09-10", "end": "2026-09-11",'
    '"why": "גישה לחוף"}], "empty": null}'
)


class _StreamChat:
    def __init__(self, raw: str, *, width: int = 6) -> None:
        self.raw = raw
        self.width = width

    def stream(self, messages, **kwargs):
        assert kwargs.get("stream_usage") is True
        assert "ליד הים" in messages[1].content
        for i in range(0, len(self.raw), self.width):
            yield SimpleNamespace(content=self.raw[i : i + self.width])
        yield SimpleNamespace(
            content="",
            usage_metadata={"input_tokens": 10, "output_tokens": 4},
        )


def test_stream_paints_spoken_text_before_json_closes():
    seen: list[str] = []
    with listen_recommend_text(seen.append):
        result = recommend_from_payload(
            "ליד הים",
            {"constraints": {}, "fits": [_fit()]},
            chat=_StreamChat(_JSON),
        )
    assert len(result.recommendations) == 1
    assert "גישה לחוף" in result.text
    assert any("אכזיב צפון" in text for text in seen)
    assert any("גישה לחוף" in text for text in seen)
    assert not any(text.lstrip().startswith("{") for text in seen)
    assert seen[-1] == result.text


def test_stream_records_usage_from_trailing_chunk():
    with collect_llm_usage() as usage:
        recommend_from_payload(
            "ליד הים",
            {"constraints": {}, "fits": [_fit()]},
            chat=_StreamChat(_JSON),
        )
    assert usage.input_tokens == 10
    assert usage.output_tokens == 4
    roles = {bucket.role: bucket for bucket in usage.by_role()}
    assert roles["recommend"].prompt_tokens == 10


def test_stream_empty_followup_paints_as_it_arrives():
    raw = '{"recommendations": [], "empty": "מתי אתם רוצים?"}'
    seen: list[str] = []
    with listen_recommend_text(seen.append):
        result = recommend_from_payload(
            "ליד הים",
            {"constraints": {}, "fits": []},
            chat=_StreamChat(raw, width=4),
        )
    assert result.recommendations == ()
    assert "מתי אתם רוצים?" in result.text
    assert any("מתי" in text for text in seen)
    assert not any('"recommendations"' in text for text in seen)
