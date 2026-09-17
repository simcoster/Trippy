"""Nebius chat retries DNS/connect blips instead of aborting the scrape."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError

from source.scraper.amenity_enrichment.llm import nebius_chat_create


def _client(create):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_nebius_chat_create_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(
        "source.scraper.amenity_enrichment.llm.time.sleep", lambda _s: None
    )
    n = {"calls": 0}

    def create(**_kwargs):
        n["calls"] += 1
        if n["calls"] < 3:
            raise APIConnectionError(request=MagicMock())
        return SimpleNamespace(ok=True)

    result = nebius_chat_create(_client(create), model="x")
    assert result.ok is True
    assert n["calls"] == 3


def test_nebius_chat_create_raises_after_retries(monkeypatch):
    monkeypatch.setattr(
        "source.scraper.amenity_enrichment.llm.time.sleep", lambda _s: None
    )

    def create(**_kwargs):
        raise APIConnectionError(request=MagicMock())

    with pytest.raises(APIConnectionError):
        nebius_chat_create(_client(create), model="x")
