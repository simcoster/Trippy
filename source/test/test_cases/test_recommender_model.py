"""Recommender defaults to Kimi-K3; Super and 235B stay opt-in."""

from source.agent.recommender import recommender_model
from source.scraper.amenity_enrichment.llm import (
    KIMI_K3_MODEL,
    NEMOTRON_SUPER_MODEL,
    QWEN_INSTRUCT_MODEL,
    chat_usd_per_mtok,
)


def test_recommender_model_default_is_kimi(monkeypatch):
    monkeypatch.delenv("TRIPPY_RECOMMENDER_MODEL", raising=False)
    assert recommender_model() == KIMI_K3_MODEL


def test_recommender_model_kimi_alias(monkeypatch):
    monkeypatch.setenv("TRIPPY_RECOMMENDER_MODEL", "kimi")
    assert recommender_model() == KIMI_K3_MODEL


def test_recommender_model_235b_alias(monkeypatch):
    monkeypatch.setenv("TRIPPY_RECOMMENDER_MODEL", "235B")
    assert recommender_model() == QWEN_INSTRUCT_MODEL


def test_recommender_model_super_alias(monkeypatch):
    monkeypatch.setenv("TRIPPY_RECOMMENDER_MODEL", "super")
    assert recommender_model() == NEMOTRON_SUPER_MODEL


def test_kimi_k3_chat_price():
    usd_in, usd_out = chat_usd_per_mtok(KIMI_K3_MODEL)
    assert usd_in == 3.00
    assert usd_out == 15.00


def test_nemotron_super_chat_price():
    usd_in, usd_out = chat_usd_per_mtok(NEMOTRON_SUPER_MODEL)
    assert usd_in == 0.30
    assert usd_out == 0.90
