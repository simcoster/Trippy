"""Timed-out / retried Nebius calls still land on LlmUsage."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError, APITimeoutError

from source.scraper.amenity_enrichment.llm import (
    LlmUsage,
    _estimated_prompt_tokens,
    nebius_chat_create,
)


def _client(create):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "x" * 40},
        {"role": "user", "content": "y" * 40},
    ]


def test_estimated_prompt_tokens_is_chars_over_four():
    assert _estimated_prompt_tokens({"messages": _messages()}) == 20


def test_timeout_then_success_counts_both_attempts(monkeypatch):
    monkeypatch.setattr(
        "source.scraper.amenity_enrichment.llm.time.sleep", lambda _s: None
    )
    n = {"calls": 0}

    def create(**_kwargs):
        n["calls"] += 1
        if n["calls"] == 1:
            raise APITimeoutError(request=MagicMock())
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3)
        )

    usage = LlmUsage()
    nebius_chat_create(
        _client(create),
        usage=usage,
        role="price_function_compile",
        model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        messages=_messages(),
    )
    assert usage.chat_calls == 2
    assert usage.chat_prompt_tokens == 20 + 12
    assert usage.chat_completion_tokens == 3
    assert usage.by_role()[0].role == "price_function_compile"


def test_exhausted_retries_still_bill_each_attempt(monkeypatch):
    monkeypatch.setattr(
        "source.scraper.amenity_enrichment.llm.time.sleep", lambda _s: None
    )

    def create(**_kwargs):
        raise APIConnectionError(request=MagicMock())

    usage = LlmUsage()
    with pytest.raises(APIConnectionError):
        nebius_chat_create(
            _client(create),
            usage=usage,
            role="listing_match",
            model="m",
            messages=_messages(),
        )
    assert usage.chat_calls == 4
    assert usage.chat_prompt_tokens == 80
