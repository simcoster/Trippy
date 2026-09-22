"""Kimi recommend falls back to Nemotron Super if the first token is late."""

from __future__ import annotations

import time
from types import SimpleNamespace

from source.agent.recommender.fallback import (
    DEFAULT_KIMI_FIRST_TOKEN_SEC,
    kimi_first_token_sec,
    kimi_super_fallback,
    primary_recommend_call,
)
from source.agent.recommender.models import recommender_model
from source.agent.recommender.recommend import recommend_from_payload
from source.agent.recommender.stream import RecommendCall
from source.agent.recommender.timing import last_recommend_timing
from source.scraper.amenity_enrichment.llm import (
    KIMI_K3_MODEL,
    NEMOTRON_SUPER_MODEL,
)

_JSON = '{"recommendations": [], "empty": "מתי?", "intro": null}'
_KIMI_JSON = '{"recommendations": [], "empty": "kimi-late", "intro": null}'


class _StreamChat:
    def __init__(self, raw: str, seen: list[str]) -> None:
        self.raw = raw
        self.seen = seen

    def stream(self, messages, **kwargs):
        self.seen.append(messages[0].content)
        yield SimpleNamespace(content=self.raw)


class _LateChat:
    def __init__(self, delay_s: float, seen: list[str]) -> None:
        self.delay_s = delay_s
        self.seen = seen

    def stream(self, messages, **kwargs):
        self.seen.append("kimi")
        time.sleep(self.delay_s)
        yield SimpleNamespace(content=_KIMI_JSON)


def test_kimi_first_token_default_is_ten_seconds(monkeypatch):
    monkeypatch.delenv("TRIPPY_KIMI_TTFT_SEC", raising=False)
    assert kimi_first_token_sec() == DEFAULT_KIMI_FIRST_TOKEN_SEC


def test_injected_chat_does_not_arm_super_fallback(monkeypatch):
    monkeypatch.setenv("TRIPPY_RECOMMENDER_MODEL", "kimi")
    primary = primary_recommend_call(
        recommender_model(), _StreamChat(_JSON, [])
    )
    assert primary.model == KIMI_K3_MODEL
    assert primary.first_token_sec == 0.0
    assert kimi_super_fallback(primary) is None


def test_kimi_ttft_timeout_falls_back_to_super(monkeypatch):
    kimi_seen: list[str] = []
    super_seen: list[str] = []
    monkeypatch.setattr(
        "source.agent.recommender.recommend.primary_recommend_call",
        lambda model, chat: RecommendCall(
            KIMI_K3_MODEL, _LateChat(0.4, kimi_seen), 0.05
        ),
    )
    monkeypatch.setattr(
        "source.agent.recommender.recommend.kimi_super_fallback",
        lambda primary: RecommendCall(
            NEMOTRON_SUPER_MODEL, _StreamChat(_JSON, super_seen)
        ),
    )
    result = recommend_from_payload(
        "ליד הים", {"constraints": {}, "fits": []}
    )
    assert kimi_seen == ["kimi"]
    assert super_seen
    assert "/no_think" in super_seen[0]
    assert "מתי?" in result.text
    assert "kimi-late" not in result.text
    timing = last_recommend_timing()
    assert timing is not None
    assert timing["model"] == NEMOTRON_SUPER_MODEL
    assert timing["fallback_from"] == KIMI_K3_MODEL


def test_kimi_fast_token_does_not_fallback(monkeypatch):
    kimi_seen: list[str] = []
    super_seen: list[str] = []
    monkeypatch.setattr(
        "source.agent.recommender.recommend.primary_recommend_call",
        lambda model, chat: RecommendCall(
            KIMI_K3_MODEL, _StreamChat(_JSON, kimi_seen), 0.4
        ),
    )
    monkeypatch.setattr(
        "source.agent.recommender.recommend.kimi_super_fallback",
        lambda primary: RecommendCall(
            NEMOTRON_SUPER_MODEL, _StreamChat(_KIMI_JSON, super_seen)
        ),
    )
    result = recommend_from_payload(
        "ליד הים", {"constraints": {}, "fits": []}
    )
    assert kimi_seen
    assert super_seen == []
    assert "מתי?" in result.text
    timing = last_recommend_timing()
    assert timing is not None
    assert timing["fallback_from"] is None
