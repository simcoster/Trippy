"""Opt-in batched judge and GLM judge model. Defaults stay per-job 235B."""

from source.agent.claim_judge import judge_batch, judge_model
from source.scraper.amenity_enrichment.llm import (
    GLM_INSTRUCT_MODEL,
    QWEN_INSTRUCT_MODEL,
    chat_usd_per_mtok,
)


def test_judge_batch_default_off(monkeypatch):
    monkeypatch.delenv("TRIPPY_JUDGE_BATCH", raising=False)
    assert judge_batch() is False


def test_judge_batch_env_on(monkeypatch):
    monkeypatch.setenv("TRIPPY_JUDGE_BATCH", "1")
    assert judge_batch() is True


def test_judge_model_glm_alias(monkeypatch):
    monkeypatch.setenv("TRIPPY_JUDGE_MODEL", "glm")
    assert judge_model() == GLM_INSTRUCT_MODEL


def test_judge_model_default_is_instruct(monkeypatch):
    monkeypatch.delenv("TRIPPY_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("TRIPPY_INSTRUCT_MODEL", raising=False)
    assert judge_model() == QWEN_INSTRUCT_MODEL


def test_glm_chat_price():
    usd_in, usd_out = chat_usd_per_mtok(GLM_INSTRUCT_MODEL)
    assert usd_in == 1.40
    assert usd_out == 4.40
