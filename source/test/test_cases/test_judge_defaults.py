"""Claim-judge defaults: compact on, five parallel live calls."""

from source.agent.claim_judge import judge_compact, judge_concurrency


def test_judge_concurrency_default_is_5(monkeypatch):
    monkeypatch.delenv("TRIPPY_JUDGE_CONCURRENCY", raising=False)
    assert judge_concurrency() == 5


def test_judge_compact_env_off(monkeypatch):
    monkeypatch.setenv("TRIPPY_JUDGE_COMPACT", "0")
    assert judge_compact() is False
